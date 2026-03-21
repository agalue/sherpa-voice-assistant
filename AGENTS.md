# AI Agent Instructions: Voice Assistant

Real-time voice assistant (Go + Rust) for edge devices. Pipeline: Mic → VAD+Whisper → Ollama → Kokoro → Speaker.

## Core Architecture

Dual implementations share models (`~/.voice-assistant/models/`) with identical behavior.

### Critical Patterns

**Lock-Free Audio**: Audio callbacks run on high-priority OS threads. **Never block or hold locks in audio callbacks.**
- **Rust**: `HeapRb` ring buffer → channel → VAD. **Go**: `sync.Pool` buffers + non-blocking `select/default`

**Separate VAD/Transcription Locks**: VAD fast (<10ms), Whisper slow (100-500ms). Use independent mutexes to prevent glitches.
- **Rust**: `Arc<Mutex<VadState>>` + `Mutex<WhisperRecognizer>`. **Go**: `atomic.Value` + `sync.RWMutex`

**Shutdown**: `running` flag = temporary pause, `shutdown` flag = permanent exit with timeout-based joins

**Hardware Acceleration**: Auto-detected at runtime. Never hardcode providers.
- **macOS**: `coreml` (ANE). **Linux + NVIDIA**: `cuda` (1 thread). **Linux CPU**: `cpu` (cores/3 threads)

## Build & File Structure

```bash
./scripts/build.sh                    # Go (auto-detects CoreML/CUDA)
cd rust-impl && ./scripts/build.sh    # Rust (auto-detects)
cd rust-impl && cargo clippy --release # Must have zero warnings
```

**Key Directories**: `internal/{audio,stt,tts,llm,config,setup}` (Go), `rust-impl/src/{audio,stt,tts,llm,config,setup}` (Rust)

## Code & Documentation

**Go**: Use `defer` for cleanup, `sync.Pool` for allocations, `context.Context` for cancellation, non-blocking `select/default`, `atomic.Value`/`sync.RWMutex`.

**Rust**: Never `unwrap()`/`expect()`/`unsafe` without justification. Use `parking_lot::Mutex`, `Arc<AtomicBool>`, lock-free structures (`ringbuf`) for audio.

**Logging**: `info!` = user events, `debug!` = internal state, `warn!` = recoverable issues

**LLM Prompt Must Include**: "Your responses will be read aloud - NEVER use markdown, asterisks, underscores, backticks, brackets, code blocks, bullet points"

**Documentation**:
- **Go**: `// Package`, `// FuncName`, document exported structs/fields/constants
- **Rust**: `//!` modules, `///` functions with `# Arguments`/`# Returns`/`# Errors`, document public items
- **Inline comments**: Only for complex algorithms, critical invariants (e.g., "never block audio callback"), workarounds. Never state the obvious.

## Testing

**Write professional tests that verify critical behavior and prevent regressions.**

**Test**:
- Core algorithms (VAD, resampling, HTML parsing)
- Data transformations with edge cases (URL decoding, entity unescaping)
- Integration points with external data (parsing real DDG HTML)
- Failure modes (empty, malformed input)

**Don't Test**:
- Trivial getters/setters
- Third-party libraries
- Language built-ins
- Config struct initialization

**Good Test Characteristics**:
- Uses real production data (actual HTML, audio samples)
- Tests behavior not implementation
- Covers edge cases (empty, max size, malformed)
- Fast (<100ms), deterministic, no network calls
- Descriptive names that read like specifications

**Organization**: Go = `*_test.go` files. Rust = `#[cfg(test)] mod tests` at file bottom.

**Naming**: Go = `TestFunctionNameEdgeCase`. Rust = `test_function_name_edge_case`.

## Key Principles

1. **No locks in audio callbacks** - use lock-free structures
2. **Separate fast/slow operations** - VAD ≠ transcription locks
3. **Timeout all thread joins** - prevent shutdown hangs
4. **Auto-detect hardware** - provider/threads adjust to platform
5. **Maintain Go/Rust parity** - identical behavior, language-specific optimizations
6. **Document thoroughly** - prioritize clarity in `cargo doc` and `go doc` output
7. **Keep READMEs current** - update project structure, architecture diagrams, and feature descriptions in both `README.md` files whenever file/module/package structure or architecture changes
