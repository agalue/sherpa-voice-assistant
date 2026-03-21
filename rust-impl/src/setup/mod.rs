//! Model download/extraction helpers and `--setup` orchestration.
//!
//! Provides [`download_file`], [`extract_tar_bz2_selected`], and
//! [`extract_tar_bz2_dir`] — used by STT and TTS model providers to
//! bootstrap model files — plus [`run_setup`], the top-level entry point
//! for the `--setup` CLI flag.

use std::fs;
use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};
use tracing::{error, info};

use crate::config::AppConfig;
use crate::stt::{self, ModelProvider as SttModelProvider, SileroModelProvider};
use crate::tts;

/// Download `url` and save it to `dest`.
///
/// If `force` is `false` and `dest` already exists the download is skipped.
///
/// Uses a blocking HTTP client suitable for calling from non-async contexts (the
/// model-setup path runs before the Tokio runtime is started).
///
/// # Errors
/// Returns an error if the HTTP request fails, the server returns a non-200
/// status, or the file cannot be written.
pub fn download_file(url: &str, dest: &Path, force: bool) -> Result<()> {
    if !force && dest.exists() {
        info!("[download] {} already exists, skipping", dest.file_name().unwrap_or_default().to_string_lossy());
        return Ok(());
    }

    if let Some(parent) = dest.parent() {
        fs::create_dir_all(parent)?;
    }

    info!("[download] Downloading {} …", dest.file_name().unwrap_or_default().to_string_lossy());

    let response = reqwest::blocking::get(url).with_context(|| format!("GET {url}"))?;
    if !response.status().is_success() {
        anyhow::bail!("GET {url}: unexpected status {}", response.status());
    }

    let bytes = response.bytes().with_context(|| format!("reading response from {url}"))?;

    // Write atomically via a temp file in the same directory.
    let tmp_path = dest.with_extension("tmp");
    let mut tmp = fs::File::create(&tmp_path).with_context(|| format!("creating {}", tmp_path.display()))?;
    tmp.write_all(&bytes)?;
    drop(tmp);
    fs::rename(&tmp_path, dest).with_context(|| format!("renaming {} → {}", tmp_path.display(), dest.display()))?;

    Ok(())
}

/// Fetch a tar.bz2 archive from `url` and extract **only** the entries listed in
/// `want_files` (a slice of `(archive_path, destination_path)` pairs).
///
/// # Errors
/// Returns an error if the HTTP request fails, the server returns a non-200 status,
/// the archive is malformed, or a file cannot be written.
pub fn extract_tar_bz2_selected(url: &str, want_files: &[(String, std::path::PathBuf)]) -> Result<()> {
    info!("[download] Fetching archive {} …", url);

    let response = reqwest::blocking::get(url).with_context(|| format!("GET {url}"))?;
    if !response.status().is_success() {
        anyhow::bail!("GET {url}: unexpected status {}", response.status());
    }

    let bytes = response.bytes()?;
    let cursor = std::io::Cursor::new(bytes);
    let bzr = bzip2::read::BzDecoder::new(cursor);
    let mut archive = tar::Archive::new(bzr);

    let mut remaining: std::collections::HashMap<&str, &std::path::Path> = want_files.iter().map(|(k, v)| (k.as_str(), v.as_path())).collect();

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();
        let path_str = path.to_string_lossy();

        if let Some(dest) = remaining.get(path_str.as_ref()).copied() {
            if let Some(parent) = dest.parent() {
                fs::create_dir_all(parent)?;
            }
            entry.unpack(dest).with_context(|| format!("extracting {} to {}", path_str, dest.display()))?;
            let key = path_str.into_owned();
            remaining.remove(key.as_str());
            if remaining.is_empty() {
                break;
            }
        }
    }

    if !remaining.is_empty() {
        for k in remaining.keys() {
            tracing::warn!("[download] Expected file not found in archive: {}", k);
        }
        anyhow::bail!("Archive did not contain all expected files");
    }

    Ok(())
}

/// Fetch a tar.bz2 archive from `url` and extract **all** entries into `dest_dir`,
/// stripping the top-level directory component from each entry path.
///
/// This is suitable for archives that contain a single top-level directory (e.g.
/// `kokoro-multi-lang-v1_0/`) whose contents should land directly in `dest_dir`.
///
/// # Errors
/// Returns an error if the HTTP request fails, the server returns a non-200 status,
/// the archive is malformed, or a file cannot be written.
pub fn extract_tar_bz2_dir(url: &str, dest_dir: &Path) -> Result<()> {
    fs::create_dir_all(dest_dir)?;

    info!("[download] Fetching archive {} …", url);

    let response = reqwest::blocking::get(url).with_context(|| format!("GET {url}"))?;
    if !response.status().is_success() {
        anyhow::bail!("GET {url}: unexpected status {}", response.status());
    }

    let bytes = response.bytes()?;
    let cursor = std::io::Cursor::new(bytes);
    let bzr = bzip2::read::BzDecoder::new(cursor);
    let mut archive = tar::Archive::new(bzr);

    for entry in archive.entries()? {
        let mut entry = entry?;
        let path = entry.path()?.into_owned();

        // Strip the leading directory component.
        let mut components = path.components();
        components.next(); // skip top-level dir
        let rel: std::path::PathBuf = components.collect();
        if rel.as_os_str().is_empty() {
            continue; // was the top-level directory entry itself
        }

        let dest = dest_dir.join(&rel);

        match entry.header().entry_type() {
            tar::EntryType::Directory => {
                fs::create_dir_all(&dest)?;
            }
            tar::EntryType::Regular => {
                if let Some(parent) = dest.parent() {
                    fs::create_dir_all(parent)?;
                }
                entry.unpack(&dest).with_context(|| format!("extracting {}", dest.display()))?;
            }
            _ => {} // ignore symlinks etc.
        }
    }

    Ok(())
}

/// Download all required model files for the configured STT and TTS backends.
///
/// This is the `--setup` mode: fetch every model archive, extract the needed
/// files, and verify that nothing is missing before the user can start the
/// assistant.
///
/// # Errors
/// Returns an error if any download, extraction, or verification step fails.
pub fn run_setup(config: &AppConfig) -> Result<()> {
    info!("🔧 Voice Assistant Setup — downloading model files");
    info!("   Model directory: {}", config.model_dir.display());
    if config.force {
        info!("   Mode: force re-download");
    } else {
        info!("   Mode: skip existing files");
    }

    let silero_provider = SileroModelProvider;
    let stt_provider = stt::new_model_provider(config)?;
    let tts_provider = tts::new_model_provider(config)?;

    info!("📥 [VAD] {} — downloading models…", silero_provider.name());
    silero_provider.ensure_models(&config.model_dir, config.force)?;

    info!("📥 [STT] {} — downloading models…", stt_provider.name());
    stt_provider.ensure_models(&config.model_dir, config.force)?;

    info!("📥 [TTS] {} — downloading models…", tts_provider.name());
    tts_provider.ensure_models(&config.model_dir, config.force)?;

    // Final verification
    info!("🔍 Verifying model files…");
    let mut all_missing: Vec<std::path::PathBuf> = Vec::new();
    all_missing.extend(silero_provider.verify_models(&config.model_dir));
    all_missing.extend(stt_provider.verify_models(&config.model_dir));
    all_missing.extend(tts_provider.verify_models(&config.model_dir));

    if !all_missing.is_empty() {
        error!("❌ Some model files are still missing:");
        for f in &all_missing {
            error!("   - {}", f.display());
        }
        anyhow::bail!("{} model file(s) missing after setup", all_missing.len());
    }

    info!("✅ All model files are present. Run the assistant without --setup to start.");
    Ok(())
}
