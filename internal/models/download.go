// Package models provides shared HTTP download and tar.bz2 extraction helpers
// used by STT and TTS [ModelProvider] implementations.
//
// Using stdlib only (net/http, archive/tar, compress/bzip2) to avoid adding
// dependencies for what is an infrequently executed code path.
package models

import (
	"archive/tar"
	"compress/bzip2"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strings"
)

// FileExists reports whether a regular file exists at path.
func FileExists(path string) bool {
	info, err := os.Stat(path)
	return err == nil && !info.IsDir()
}

// DownloadFile fetches url and saves it to dest.
// If force is false and dest already exists, the download is skipped.
func DownloadFile(url, dest string, force bool) error {
	if !force && FileExists(dest) {
		log.Printf("[download] %s already exists, skipping", filepath.Base(dest))
		return nil
	}

	if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
		return err
	}

	tmp, err := os.CreateTemp(filepath.Dir(dest), ".download-*")
	if err != nil {
		return err
	}
	tmpName := tmp.Name()
	defer func() { _ = os.Remove(tmpName) }() // clean up on failure

	log.Printf("[download] Downloading %s …", filepath.Base(dest))
	resp, err := http.Get(url) //nolint:gosec // URL is constructed from hardcoded base + model name
	if err != nil {
		tmp.Close()
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		tmp.Close()
		return fmt.Errorf("GET %s: unexpected status %s", url, resp.Status)
	}

	if _, err := io.Copy(tmp, resp.Body); err != nil {
		tmp.Close()
		return fmt.Errorf("writing %s: %w", dest, err)
	}
	if err := tmp.Close(); err != nil {
		return err
	}

	return os.Rename(tmpName, dest)
}

// ExtractTarBz2Selected fetches a tar.bz2 archive from url and extracts only the
// entries whose tar path is a key in wantFiles; the value maps to the destination
// file path on disk.
func ExtractTarBz2Selected(url string, wantFiles map[string]string) error {
	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: unexpected status %s", url, resp.Status)
	}

	bzr := bzip2.NewReader(resp.Body)
	tr := tar.NewReader(bzr)

	remaining := make(map[string]string, len(wantFiles))
	for k, v := range wantFiles {
		remaining[k] = v
	}

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("reading archive: %w", err)
		}

		dest, ok := remaining[hdr.Name]
		if !ok {
			continue
		}

		if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
			return err
		}

		f, err := os.Create(dest)
		if err != nil {
			return err
		}
		if _, err := io.Copy(f, tr); err != nil { //nolint:gosec // controlled archive entry size
			f.Close()
			return fmt.Errorf("extracting %s: %w", hdr.Name, err)
		}
		f.Close()

		delete(remaining, hdr.Name)
		if len(remaining) == 0 {
			break // all wanted files extracted
		}
	}

	if len(remaining) > 0 {
		for k := range remaining {
			log.Printf("[download] WARNING: expected file not found in archive: %s", k)
		}
		return fmt.Errorf("archive did not contain all expected files")
	}

	return nil
}

// ExtractTarBz2Dir fetches a tar.bz2 archive from url and extracts all entries
// into destDir, stripping the top-level directory component from each path.
// This is useful for archives that contain a single top-level directory (e.g.
// kokoro-multi-lang-v1_0/) whose contents should land directly in destDir.
func ExtractTarBz2Dir(url, destDir string) error {
	if err := os.MkdirAll(destDir, 0o755); err != nil {
		return err
	}

	resp, err := http.Get(url) //nolint:gosec
	if err != nil {
		return fmt.Errorf("GET %s: %w", url, err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("GET %s: unexpected status %s", url, resp.Status)
	}

	bzr := bzip2.NewReader(resp.Body)
	tr := tar.NewReader(bzr)

	for {
		hdr, err := tr.Next()
		if err == io.EOF {
			break
		}
		if err != nil {
			return fmt.Errorf("reading archive: %w", err)
		}

		// Strip the leading directory component (e.g. "kokoro-multi-lang-v1_0/")
		parts := strings.SplitN(hdr.Name, "/", 2)
		if len(parts) < 2 || parts[1] == "" {
			continue // top-level directory entry itself
		}
		rel := parts[1]

		dest := filepath.Join(destDir, filepath.FromSlash(rel))

		switch hdr.Typeflag {
		case tar.TypeDir:
			if err := os.MkdirAll(dest, 0o755); err != nil {
				return err
			}
		case tar.TypeReg:
			if err := os.MkdirAll(filepath.Dir(dest), 0o755); err != nil {
				return err
			}
			f, err := os.Create(dest)
			if err != nil {
				return err
			}
			if _, err := io.Copy(f, tr); err != nil { //nolint:gosec
				f.Close()
				return fmt.Errorf("extracting %s: %w", hdr.Name, err)
			}
			f.Close()
		}
	}

	return nil
}
