# Jetson Orin Nano Optimization Guide

Running the voice assistant on Jetson Orin Nano Super (8GB) with optimized memory usage.

## Quick Setup

```bash
# Automated configuration (recommended)
./scripts/jetson_setup.sh setup

# Manual status check
./scripts/jetson_setup.sh status
```

## Known Limitations

**⚠️ Tool Calling Accuracy**  
Due to memory constraints, Jetson users must use smaller models (1.5b, 3b parameters). These models have **reduced accuracy** when using agentic tools (weather, web search) compared to 7B models. The assistant may:
- Not recognize when to call a tool
- Call tools unnecessarily
- Format tool parameters incorrectly

For best tool calling performance, 7B models are recommended, but these require significantly more memory (~4.9GB) and may not be feasible on 8GB devices running the full pipeline.

## Manual Configuration

### 1. Configure Ollama

```bash
# Edit systemd service
sudo systemctl edit ollama
```

Add these environment variables:

```ini
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_CTX=1024"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_MAX_VRAM=3221225472"
```

Apply changes:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

**What this does:** 
- Limits memory usage by reducing context window (1024 tokens)
- Keeps model loaded indefinitely (`-1`) - no reload delays in conversations
- Limits Ollama to 3GB max (VRAM) on Jetson's unified memory
- Handles one request at a time to prevent memory spikes
- All inference on GPU (CUDA) for maximum performance

### 2. Pull Optimized Model

```bash
# Q2_K quantization - recommended for Jetson (~1.2GB)
ollama pull qwen2.5:3b-instruct-q2_k

# Q3_K_M quantization - better quality but more memory (~1.5GB)
ollama pull qwen2.5:3b-instruct-q3_k_m
```

### 3. Start SearXNG

```bash
cd searxng
sudo docker compose up -d
```

**Note:** Docker commands on Jetson require `sudo` by default.

Configuration files (`docker-compose.yml`, `settings.yml`) are pre-configured for minimal resource usage (384MB RAM, 1 CPU core, Bing search).

## Running the Assistant

### Automatic Pre-loading (Recommended)

The `run-voice-assistant.sh` script automatically detects Jetson and pre-loads the Ollama model:

```bash
# Go: Default qwen2.5:1.5b with tiny Whisper
./run-voice-assistant.sh -ollama-model qwen2.5:1.5b -whisper-model tiny

# Rust: Default qwen2.5:1.5b with tiny Whisper
./run-voice-assistant.sh --ollama-model qwen2.5:1.5b --whisper-model tiny

# Or use a different model
./run-voice-assistant.sh -ollama-model qwen2.5:3b-instruct-q2_k -whisper-model tiny
```

**What happens automatically on Jetson:**
1. Detects Jetson hardware (`/etc/nv_tegra_release`)
2. Extracts model name from command line args
3. Pre-loads model with `num_ctx=1024` to reserve GPU memory
4. Starts voice assistant

**Why pre-loading is required:** On Jetson's unified memory, loading voice assistant first fragments memory, making large contiguous allocations fail. Pre-loading the LLM reserves the largest block first.

### Larger Models (More Capable)

For better quality responses with more memory:

```bash
# 3b model q2_k quantization (~1.2GB)
./run-voice-assistant.sh -ollama-model qwen2.5:3b-instruct-q2_k -whisper-model tiny

# 3b model q3_k_m quantization (~1.5GB)  
./run-voice-assistant.sh -ollama-model qwen2.5:3b-instruct-q3_k_m -whisper-model tiny
```

## Memory Optimization Summary

**Actual measurements on Jetson Orin Nano Super 8GB** (unified memory architecture where CPU and GPU share same RAM):

| Component | Memory | Model Size |
|-----------|---------|------------|
| Voice Assistant (Whisper small + Kokoro TTS + VAD) | **~2.6GB** | Default configuration |
| Voice Assistant (Whisper tiny + Kokoro TTS + VAD) | **~1.5GB** | **⭐ Recommended for Jetson** |
| Qwen q2_k | ~1.2GB | **⭐ Recommended for Jetson** |
| Qwen q3_k_m | ~1.5GB | Better quality, more memory |
| SearXNG (Docker) | ~384MB | Optional web search |
| System overhead | ~500MB | OS, buffers, other processes |

**Why Whisper tiny is critical:** On Jetson's unified memory, CUDA has allocation limits even when free memory is available. Using Whisper small (~2.6GB) leaves only ~5GB for other processes, causing CUDA allocation failures when Ollama tries to load the LLM model. Whisper tiny reduces this to ~1.5GB, leaving ~6.5GB available.

**Expected total with recommended config:**
- Whisper tiny + Kokoro TTS: ~1.5GB
- Qwen q2_k: ~1.2GB  
- SearXNG: ~0.4GB
- System: ~0.5GB
- **Total: ~3.6GB used, ~4.4GB free**

## Additional Optimizations

### Increase Swap (if needed)

```bash
sudo fallocate -l 8G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab
```

### Set Maximum Performance

```bash
sudo nvpmodel -m 0
sudo jetson_clocks
```

## Troubleshooting

### Out of Memory Errors

**Symptom**: `cudaMalloc failed: out of memory` or `unable to allocate CUDA0 buffer` when you speak to the assistant

**Root cause**: Jetson's unified memory architecture means CPU and GPU share the same 8GB RAM. Starting voice assistant first fragments memory, preventing Ollama from allocating large contiguous blocks.

**Solution: Use `run-voice-assistant.sh` which auto-detects Jetson and pre-loads**

```bash
./run-voice-assistant.sh -ollama-model qwen2.5:1.5b -whisper-model tiny
```

The script automatically handles pre-loading on Jetson. If you see errors:

1. **Verify model is pulled:**
   ```bash
   ollama list
   ollama pull qwen2.5:1.5b  # If not listed
   ```

2. **Check Ollama is running:**
   ```bash
   curl http://localhost:11434/api/version
   sudo systemctl status ollama
   ```

3. **Monitor memory during startup:**
   ```bash
   jtop  # Watch GPU MEM while starting voice assistant
   ```

### NvMapMemAlloc Warnings During TTS

**Symptom**: `NvMapMemAllocInternalTagged: 1075072515 error 12` messages during speech synthesis

**Impact**: **Non-fatal** - audio still plays correctly. These are memory allocation warnings from GPU trying to allocate small buffers in fragmented memory.

**Why it happens**: After LLM and Whisper allocate large blocks, GPU memory is fragmented. TTS tries to allocate small buffers but kernel warns about fragmentation.

**Solution**: Cosmetic only - no action needed. Audio synthesis works despite warnings. If concerned, restart voice assistant to defragment memory.

### General Out of Memory

```bash
# Restart services
sudo systemctl restart ollama
cd searxng && sudo docker compose restart

# Check memory
free -h
sudo tegrastats
```

### Ollama Not Responding

```bash
sudo systemctl status ollama
sudo journalctl -u ollama -n 50
sudo systemctl restart ollama
```

### SearXNG Issues

```bash
cd searxng
sudo docker compose logs
sudo docker compose restart
```

## Model Quantization Options

For GPU-only inference on Jetson unified memory:

| Model | Size | Quality | Speed | Recommendation |
|-------|------|---------|-------|----------------|
| qwen2.5:3b-q2_k | ~1.2GB | Good | Fast | ⭐ **Recommended** |
| qwen2.5:3b-q3_k_m | ~1.5GB | Better | Medium | If memory allows |
| qwen2.5:3b-q4_0 | ~2.0GB | Best | Slower | May cause OOM |
| qwen2.5:1.5b-q4_0 | ~1.0GB | Lower | Fallback option |

## Useful Commands

```bash
# Management script
./scripts/jetson_setup.sh status           # Show status
./scripts/jetson_setup.sh logs [service]   # View logs
./scripts/jetson_setup.sh restart          # Restart all

# Direct commands
ollama list                                # List models
sudo systemctl cat ollama                  # Show config
docker stats searxng                       # Monitor SearXNG
```
