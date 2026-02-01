# AI Agent Instructions: Voice Assistant

Real-time voice assistant (Go + Rust) for edge devices. Pipeline: Mic → VAD+Whisper → Ollama → Kokoro → Speaker.

## Core Architecture

**Dual implementations share models** (`~/.voice-assistant/models/`) with identical behavior but language-specific optimizations.

### Critical Pattern: Lock-Free Audio
Audio callbacks run on high-priority OS threads. **Never block or hold locks in audio callbacks.**

**Rust** ([src/audio/capture.rs](rust-impl/src/audio/capture.rs)): Lock-free ring buffer → channel → callback thread → VAD
```rust
HeapRb::new(65536) → sync_channel(32) → vad_accept_waveform()
```

**Go** ([internal/audio/capture.go](internal/audio/capture.go)): Buffer pooling with non-blocking send
```go
buffer := float32Pool.Get().(*[]float32)
select { case audioChan <- samples: default: }  // Never block
```

### Separate VAD and Transcription Locks
VAD is fast (<10ms), Whisper is slow (100-500ms). **Use independent mutexes** to prevent audio glitches.

**Rust**: `Arc<Mutex<VadState>>` (VAD) + `Mutex<WhisperRecognizer>` (transcription)  
**Go**: `atomic.Value` for state + `sync.RWMutex` for VAD access

### Shutdown: Pause vs Stop
- `running` flag: Temporary pause (half-duplex mode during playback)
- `shutdown` flag: Permanent exit with timeout-based thread joins

```rust
self.shutdown.store(true, Ordering::SeqCst);
drop(self.sender.take());  // Wake blocked operations
std::thread::sleep(Duration::from_millis(100));
if !handle.is_finished() { warn!("Thread didn't exit"); }
```

## Hardware Acceleration

Auto-detected at runtime. Thread counts adjust based on provider:
- **macOS**: `coreml` (ANE)
- **Linux + NVIDIA**: `cuda` (1 thread - GPU handles parallelism)
- **Linux CPU**: `cpu` (cores/3 threads)

**Never hardcode providers.** See `normalize_thread_counts()` in config files.

## Build Commands

```bash
# Go (auto-detects CoreML/CUDA)
./scripts/build.sh

# Rust (auto-detects CoreML/CUDA)
cd rust-impl && ./scripts/build.sh && cd ..

# Rust - Always run clippy before committing (must have zero warnings)
cd rust-impl && cargo clippy --release & cd ..
```

## File Structure

```
internal/audio/     # Go: malgo capture/playback, sync.Pool buffers
internal/stt/       # Go: VAD + Whisper, atomic.Value state
internal/llm/       # Go: Ollama client, connection pooling
internal/config/    # Go: CLI flags, provider auto-detection

rust-impl/src/audio/ # Rust: cpal + ringbuf, lock-free capture
rust-impl/src/stt/   # Rust: Separate VAD/Whisper mutexes
rust-impl/src/llm/   # Rust: RIG framework, 60s timeout
rust-impl/src/config/ # Rust: Clap CLI (identical flags to Go)
```

## Code Guidelines

**Go Best Practices**:
- Use `defer` for cleanup (mutex unlocks, channel closes)
- Prefer `sync.Pool` for frequent allocations
- Use `context.Context` for cancellation propagation
- Buffered channels with `select/default` for non-blocking sends
- `atomic.Value` for lock-free reads, `sync.RWMutex` when writes are rare

**Rust Safety Patterns**:
- **Never use `unwrap()` or `expect()`** - use proper error handling with `?` or `match`
- **Never use `unsafe`** blocks without explicit justification in comments
- Prefer `parking_lot::Mutex` over `std::sync::Mutex` for performance
- Use `Arc<AtomicBool>` for shared flags, not `Arc<Mutex<bool>>`
- Lock-free structures (`ringbuf`) over mutexes for audio path

**Shared Conventions**:
- `info!`: User-facing events (speech started, LLM response)
- `debug!`: Internal state (buffer sizes, thread lifecycle)
- `warn!`: Recoverable issues (thread timeout, buffer overflow)

**LLM Prompt**: Must forbid markdown and emojis (responses are read aloud):
```
IMPORTANT: Your responses will be read aloud, so you must NEVER use markdown,
asterisks, underscores, backticks, brackets, code blocks, bullet points...
```

## Documentation Standards

### Go (`go doc`)
- **Package**: `// Package name` + purpose
- **Functions**: `// FuncName` + description. Add Parameters/Returns sections for complex cases
- **Structs**: Document purpose + exported fields (inline or above)
- **Constants**: Group related + explain value rationale

### Rust (`cargo doc`)
- **Module**: `//!` at top + purpose
- **Functions**: `///` + `# Arguments`, `# Returns`, `# Errors` sections
- **Structs**: `///` + document public fields with `///`
- **Constants**: `///` + explain rationale

### Inline Comments
**Use when:**
- Complex algorithms, non-obvious safety/performance
- Critical invariants (e.g., "never block audio callback")
- Workarounds for library issues

**Never:**
- Obvious code (refactor if unclear)
- Restating what code does

**Examples:**
```rust
// Audio callback runs on high-priority OS thread - never block or hold locks
if running_clone.load(Ordering::Relaxed) { ... }
```
```go
// length_scale is inverse of speed (0.5 = 2x, 2.0 = 0.5x)
ttsConfig.Model.Kokoro.LengthScale = 1.0 / cfg.Speed
```

### Checklist
- [ ] All exported/public items documented
- [ ] Struct fields documented (especially public)
- [ ] Constants explain values/purpose
- [ ] Rust `Result` functions have `# Errors`
- [ ] Inline comments justified (explain "why" not "what")

## Key Principles

1. **No locks in audio callbacks** - use lock-free structures
2. **Separate fast/slow operations** - VAD ≠ transcription locks
3. **Timeout all thread joins** - prevent shutdown hangs
4. **Auto-detect hardware** - provider/threads adjust to platform
5. **Maintain Go/Rust parity** - identical behavior, language-specific optimizations
6. **Document thoroughly** - prioritize clarity in `cargo doc` and `go doc` output
