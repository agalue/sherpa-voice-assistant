#!/bin/bash
# Setup script for the Rust Voice Assistant
# Downloads required models for speech recognition and synthesis

set -e

# Determine script and project directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Parse arguments early to handle --help before any work
FORCE_DOWNLOAD=false
ASSETS_DIR=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force)
            FORCE_DOWNLOAD=true
            shift
            ;;
        --assets-dir)
            ASSETS_DIR="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --assets-dir DIR  Installation directory (default: ~/.voice-assistant)"
            echo "  --force           Force re-download even if files exist"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Run '$0 --help' for usage information"
            exit 1
            ;;
    esac
done

# Set default assets directory if not specified
if [ -z "$ASSETS_DIR" ]; then
    ASSETS_DIR="${HOME}/.voice-assistant"
fi

MODEL_DIR="${ASSETS_DIR}/models"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check for required tools
check_dependencies() {
    local missing=()

    for cmd in curl tar; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done

    if [ ${#missing[@]} -ne 0 ]; then
        log_error "Missing required tools: ${missing[*]}"
        exit 1
    fi
}

# Create model directories
create_directories() {
    log_info "Creating model directories..."
    mkdir -p "${MODEL_DIR}/whisper"
    mkdir -p "${MODEL_DIR}/tts"
}

# Download Whisper model (small multilingual for 99 languages support)
download_whisper() {
    local WHISPER_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models"
    local WHISPER_MODEL="sherpa-onnx-whisper-small"

    if [ "$FORCE_DOWNLOAD" = false ] && \
       [ -f "${MODEL_DIR}/whisper/whisper-small-encoder.int8.onnx" ] && \
       [ -f "${MODEL_DIR}/whisper/whisper-small-decoder.int8.onnx" ] && \
       [ -f "${MODEL_DIR}/whisper/whisper-small-tokens.txt" ]; then
        log_info "Whisper multilingual model already exists, skipping..."
        return
    fi

    log_info "Downloading Whisper model (small multilingual int8 - 99 languages)..."

    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    curl -L "${WHISPER_URL}/${WHISPER_MODEL}.tar.bz2" -o "${tmp_dir}/whisper.tar.bz2"
    tar -xjf "${tmp_dir}/whisper.tar.bz2" -C "${tmp_dir}"

    # Copy int8 quantized models (smaller and faster)
    cp "${tmp_dir}/${WHISPER_MODEL}/small-encoder.int8.onnx" "${MODEL_DIR}/whisper/whisper-small-encoder.int8.onnx"
    cp "${tmp_dir}/${WHISPER_MODEL}/small-decoder.int8.onnx" "${MODEL_DIR}/whisper/whisper-small-decoder.int8.onnx"
    cp "${tmp_dir}/${WHISPER_MODEL}/small-tokens.txt" "${MODEL_DIR}/whisper/whisper-small-tokens.txt"

    log_info "Whisper multilingual model downloaded successfully"
}

# Download Silero VAD model
download_vad() {
    local VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"

    if [ "$FORCE_DOWNLOAD" = false ] && [ -f "${MODEL_DIR}/silero_vad.onnx" ]; then
        log_info "VAD model already exists, skipping..."
        return
    fi

    log_info "Downloading Silero VAD model..."
    curl -L "${VAD_URL}" -o "${MODEL_DIR}/silero_vad.onnx"
    log_info "VAD model downloaded successfully"
}

# Download Kokoro TTS model (multi-lang v1.0 - supports CoreML on macOS)
download_default_voice() {
    local KOKORO_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
    local KOKORO_DIR="${MODEL_DIR}/tts/kokoro-multi-lang-v1_0"

    if [ "$FORCE_DOWNLOAD" = false ] && \
       [ -f "${KOKORO_DIR}/model.onnx" ] && \
       [ -f "${KOKORO_DIR}/voices.bin" ] && \
       [ -f "${KOKORO_DIR}/tokens.txt" ]; then
        log_info "Kokoro TTS model already exists, skipping..."
        return
    fi

    log_info "Downloading Kokoro TTS model (kokoro-multi-lang-v1_0, ~333MB)..."

    local tmp_dir=$(mktemp -d)
    trap "rm -rf $tmp_dir" EXIT

    curl -L "${KOKORO_URL}" -o "${tmp_dir}/kokoro.tar.bz2"
    tar -xjf "${tmp_dir}/kokoro.tar.bz2" -C "${tmp_dir}"

    # Move extracted directory to tts folder
    rm -rf "${KOKORO_DIR}"
    mv "${tmp_dir}/kokoro-multi-lang-v1_0" "${KOKORO_DIR}"

    log_info "Kokoro TTS model downloaded successfully"
    log_info "Default voice: bf_emma (British female, speaker ID 21)"
}

# Download espeak-ng data if not already present from Kokoro
download_espeak_data() {
    local KOKORO_ESPEAK="${MODEL_DIR}/tts/kokoro-multi-lang-v1_0/espeak-ng-data"

    if [ -d "${KOKORO_ESPEAK}" ]; then
        log_info "espeak-ng data present in Kokoro model, skipping separate download..."
        return
    fi

    local ESPEAK_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/espeak-ng-data.tar.bz2"

    log_info "Downloading espeak-ng data..."

    curl -L "${ESPEAK_URL}" -o "/tmp/espeak-ng-data.tar.bz2"
    tar -xjf "/tmp/espeak-ng-data.tar.bz2" -C "${MODEL_DIR}/tts/kokoro-multi-lang-v1_0"
    rm "/tmp/espeak-ng-data.tar.bz2"

    log_info "espeak-ng data downloaded successfully"
}

# Check Ollama installation
check_ollama() {
    if ! command -v ollama &> /dev/null; then
        log_warn "Ollama is not installed."
        log_warn "Please install Ollama from: https://ollama.ai"
        log_warn "After installation, run: ollama pull gemma3:1b"
    else
        log_info "Ollama is installed"

        # Check if default model is available
        if ! ollama list | grep -q "gemma3:1b"; then
            log_warn "Default model (gemma3:1b) not found"
            log_warn "Run: ollama pull gemma3:1b"
        else
            log_info "Default model (gemma3:1b) is available"
        fi
    fi
}

# Print system info
print_system_info() {
    log_info "System Information:"
    echo "  OS: $(uname -s)"
    echo "  Architecture: $(uname -m)"

    # Check for GPU
    if [ "$(uname -s)" = "Darwin" ]; then
        echo "  GPU: Apple Silicon (CoreML available)"
    elif [ -e "/dev/nvidia0" ]; then
        echo "  GPU: NVIDIA (CUDA available)"
    else
        echo "  GPU: None detected (using CPU)"
    fi
}

# Main
main() {
    log_info "Voice Assistant Setup Script"
    echo ""
    echo "Installation directory: ${ASSETS_DIR}"
    echo ""

    print_system_info
    echo ""

    check_dependencies
    create_directories

    echo ""
    log_info "Downloading models..."
    echo ""

    download_whisper
    download_vad
    download_default_voice
    download_espeak_data

    echo ""
    check_ollama

    echo ""
    log_info "Setup complete!"
    echo ""
    echo "Models installed to: ${ASSETS_DIR}"
    echo ""
    echo "To run the voice assistant:"
    echo "  cd ${PROJECT_DIR}"
    echo "  cargo run --release -- --model-dir ${MODEL_DIR}"
    echo ""
    echo "Available Kokoro voices (multi-lang v1.0 - 53 speakers):"
    echo "  American: af_bella (0), af_nicole (1), af_sarah (2), af_sky (3), am_adam (4), am_michael (5)"
    echo "  British:  bf_emma (21), bf_isabella (22), bm_george (23), bm_lewis (24)"
    echo ""
    echo "To change voice, use: --tts-voice <name> --tts-speaker-id <id>"
    echo "Example: cargo run --release -- --tts-voice bf_emma --tts-speaker-id 21"
}

main "$@"
