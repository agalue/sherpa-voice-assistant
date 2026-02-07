#!/bin/bash
#
# Setup script for Voice Assistant (Go + sherpa-onnx)
# Downloads required models: Silero-VAD, Whisper, and Kokoro TTS
# Detects platform and CUDA availability for optimal configuration
#

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${GREEN}╔═══════════════════════════════════════════════════════════╗${NC}"
echo -e "${GREEN}║     Voice Assistant Setup - Model Download Script         ║${NC}"
echo -e "${GREEN}╚═══════════════════════════════════════════════════════════╝${NC}"
echo

# Check for required dependencies
check_dependencies() {
    local missing=()

    for cmd in curl tar; do
        if ! command -v "$cmd" &> /dev/null; then
            missing+=("$cmd")
        fi
    done

    if [[ ${#missing[@]} -ne 0 ]]; then
        echo -e "${RED}✗ Missing required tools: ${missing[*]}${NC}"
        echo -e "${RED}  Please install them before continuing.${NC}"
        exit 1
    fi
}

# Detect platform
detect_platform() {
    local os=$(uname -s)
    local arch=$(uname -m)

    case "$os" in
        Darwin)
            PLATFORM="macos"
            if [[ "$arch" == "arm64" ]]; then
                ARCH="apple-silicon"
                PROVIDER="coreml"
            else
                ARCH="intel"
                PROVIDER="coreml"
            fi
            ;;
        Linux)
            PLATFORM="linux"
            ARCH="$arch"
            # Check for NVIDIA GPU (discrete or Jetson SOC)
            if command -v nvidia-smi &> /dev/null || \
               [[ -e /dev/nvidia0 ]] || \
               [[ -e /dev/nvhost-gpu ]] || \
               [[ -f /etc/nv_tegra_release ]]; then
                HAS_NVIDIA=true
                PROVIDER="cuda"
            else
                HAS_NVIDIA=false
                PROVIDER="cpu"
            fi
            ;;
        *)
            PLATFORM="unknown"
            PROVIDER="cpu"
            ;;
    esac
}

# Detect CUDA version if available (discrete GPU or Jetson)
detect_cuda() {
    if [[ "$PLATFORM" != "linux" ]]; then
        return
    fi

    # Check for discrete GPU via nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n1)
        if [[ -n "$CUDA_VERSION" ]]; then
            echo -e "${GREEN}✓ NVIDIA GPU detected (driver: $CUDA_VERSION)${NC}"
            
            # Check for CUDA toolkit
            if command -v nvcc &> /dev/null; then
                NVCC_VERSION=$(nvcc --version | grep "release" | awk '{print $6}' | cut -d',' -f1)
                echo -e "${GREEN}✓ CUDA toolkit installed: $NVCC_VERSION${NC}"
            else
                echo -e "${YELLOW}⚠ CUDA toolkit not found in PATH${NC}"
                echo -e "${YELLOW}  Install with: sudo apt install nvidia-cuda-toolkit${NC}"
            fi
            return
        fi
    fi

    # Check for Jetson device
    if [[ -f "/etc/nv_tegra_release" ]]; then
        JETSON_VERSION=$(head -n1 /etc/nv_tegra_release 2>/dev/null)
        echo -e "${GREEN}✓ NVIDIA Jetson detected${NC}"
        echo -e "${GREEN}  L4T: $JETSON_VERSION${NC}"
        
        # Check JetPack version if available
        if [[ -f "/etc/apt/sources.list.d/nvidia-l4t-apt-source.list" ]]; then
            echo -e "${GREEN}✓ JetPack SDK installed${NC}"
        fi
        return
    fi

    # Check for Jetson via device tree
    if [[ -f "/proc/device-tree/compatible" ]]; then
        if grep -q "nvidia,tegra\|nvidia,jetson" /proc/device-tree/compatible 2>/dev/null; then
            echo -e "${GREEN}✓ NVIDIA Jetson/Tegra platform detected${NC}"
            return
        fi
    fi

    # Check for Jetson GPU devices
    if [[ -e "/dev/nvhost-gpu" ]] || [[ -e "/dev/nvmap" ]]; then
        echo -e "${GREEN}✓ NVIDIA Jetson GPU detected${NC}"
        return
    fi
}

check_dependencies
detect_platform
echo -e "${BLUE}Platform: ${PLATFORM} (${ARCH})${NC}"
echo -e "${BLUE}Recommended provider: ${PROVIDER}${NC}"
detect_cuda
echo

# Default installation directory
ASSETS_DIR="${HOME}/.voice-assistant"
MODELS_DIR="${ASSETS_DIR}/models"

# Force download flag
FORCE_DOWNLOAD=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --assets-dir)
            ASSETS_DIR="$2"
            MODELS_DIR="${ASSETS_DIR}/models"
            shift 2
            ;;
        --force)
            FORCE_DOWNLOAD=true
            shift
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
            echo -e "${RED}Unknown option: $1${NC}"
            exit 1
            ;;
    esac
done

echo -e "${YELLOW}Installation directory: ${ASSETS_DIR}${NC}"
echo

# Create directories
mkdir -p "${MODELS_DIR}/whisper"
mkdir -p "${MODELS_DIR}/tts"

# Base URL for sherpa-onnx releases
SHERPA_BASE="https://github.com/k2-fsa/sherpa-onnx/releases/download"

# Function to download and extract (using temp directory for safety)
download_and_extract() {
    local url="$1"
    local output_dir="$2"
    local filename=$(basename "$url")

    echo -e "${YELLOW}Downloading: ${filename}${NC}"

    if [[ "$filename" == *.tar.bz2 ]]; then
        (
            tmp_dir=$(mktemp -d)
            trap 'rm -rf "$tmp_dir"' EXIT

            curl -L "$url" -o "${tmp_dir}/${filename}"
            tar -xjf "${tmp_dir}/${filename}" -C "${tmp_dir}"

            # Move extracted content to output directory
            extracted_dir=$(find "${tmp_dir}" -mindepth 1 -maxdepth 1 -type d | head -n1)
            if [[ -n "$extracted_dir" ]]; then
                cp -r "${extracted_dir}"/* "${output_dir}/"
            fi
        )
    elif [[ "$filename" == *.onnx ]]; then
        curl -L "$url" -o "${output_dir}/${filename}"
    fi

    echo -e "${GREEN}✓ Downloaded: ${filename}${NC}"
}

# Function to check Ollama installation and model availability
check_ollama() {
    echo
    echo -e "${GREEN}Checking Ollama installation...${NC}"
    
    if ! command -v ollama &> /dev/null; then
        echo -e "${YELLOW}⚠ Ollama is not installed${NC}"
        echo -e "${YELLOW}  Install from: https://ollama.ai${NC}"
        echo -e "${YELLOW}  Then run: ollama pull gemma3:1b${NC}"
        return
    fi
    
    echo -e "${GREEN}✓ Ollama is installed${NC}"
    
    # Check if default model is available
    if ! ollama list | grep -q "gemma3:1b"; then
        echo -e "${YELLOW}⚠ Default model (gemma3:1b) not found${NC}"
        echo -e "${YELLOW}  Run: ollama pull gemma3:1b${NC}"
    else
        echo -e "${GREEN}✓ Default model (gemma3:1b) is available${NC}"
    fi
}

# 1. Download Silero VAD
echo
echo -e "${GREEN}[1/4] Downloading Silero VAD model...${NC}"
VAD_URL="${SHERPA_BASE}/asr-models/silero_vad.onnx"
if [[ "$FORCE_DOWNLOAD" = false && -f "${MODELS_DIR}/silero_vad.onnx" ]]; then
    echo -e "${YELLOW}Silero VAD already exists, skipping...${NC}"
else
    curl -L "$VAD_URL" -o "${MODELS_DIR}/silero_vad.onnx"
    echo -e "${GREEN}✓ Silero VAD model installed${NC}"
fi

# 2. Download Whisper model (multilingual small - supports 99 languages)
echo
echo -e "${GREEN}[2/4] Downloading Whisper STT model (small multilingual)...${NC}"
WHISPER_URL="${SHERPA_BASE}/asr-models/sherpa-onnx-whisper-small.tar.bz2"
WHISPER_DIR="${MODELS_DIR}/whisper"

if [[ "$FORCE_DOWNLOAD" = false && -f "${WHISPER_DIR}/whisper-small-encoder.int8.onnx" ]]; then
    echo -e "${YELLOW}Whisper model already exists, skipping...${NC}"
else
    echo -e "${YELLOW}This may take a few minutes...${NC}"
    
    tmp_dir=$(mktemp -d)
    
    curl -L "$WHISPER_URL" -o "${tmp_dir}/whisper.tar.bz2"
    tar -xjf "${tmp_dir}/whisper.tar.bz2" -C "${tmp_dir}"

    # Move only int8 quantized models (smaller, faster)
    cp "${tmp_dir}/sherpa-onnx-whisper-small/small-encoder.int8.onnx" "${WHISPER_DIR}/whisper-small-encoder.int8.onnx"
    cp "${tmp_dir}/sherpa-onnx-whisper-small/small-decoder.int8.onnx" "${WHISPER_DIR}/whisper-small-decoder.int8.onnx"
    cp "${tmp_dir}/sherpa-onnx-whisper-small/small-tokens.txt" "${WHISPER_DIR}/whisper-small-tokens.txt"
    
    rm -rf "${tmp_dir}"

    echo -e "${GREEN}✓ Whisper model installed${NC}"
fi

# 3. Download Kokoro TTS model (multi-lang v1.0 - supports CoreML on macOS)
echo
echo -e "${GREEN}[3/4] Downloading Kokoro TTS model (kokoro-multi-lang-v1_0)...${NC}"
TTS_URL="${SHERPA_BASE}/tts-models/kokoro-multi-lang-v1_0.tar.bz2"
TTS_DIR="${MODELS_DIR}/tts"
KOKORO_DIR="${TTS_DIR}/kokoro-multi-lang-v1_0"

if [[ "$FORCE_DOWNLOAD" = false && -f "${KOKORO_DIR}/model.onnx" ]]; then
    echo -e "${YELLOW}Kokoro TTS model already exists, skipping...${NC}"
else
    echo -e "${YELLOW}This may take a few minutes (~333MB download)...${NC}"
    
    tmp_dir=$(mktemp -d)
    
    curl -L "$TTS_URL" -o "${tmp_dir}/kokoro.tar.bz2"
    tar -xjf "${tmp_dir}/kokoro.tar.bz2" -C "${tmp_dir}"

    # Move extracted directory to tts folder
    rm -rf "${KOKORO_DIR}"
    mv "${tmp_dir}/kokoro-multi-lang-v1_0" "${KOKORO_DIR}"
    
    rm -rf "${tmp_dir}"

    echo -e "${GREEN}✓ Kokoro TTS model installed${NC}"
fi

# 4. Download espeak-ng data if not present in Kokoro
echo
echo -e "${GREEN}[4/4] Checking espeak-ng data...${NC}"
ESPEAK_DIR="${KOKORO_DIR}/espeak-ng-data"

if [[ -d "${ESPEAK_DIR}" ]]; then
    echo -e "${YELLOW}espeak-ng data present in Kokoro model, skipping...${NC}"
else
    echo -e "${YELLOW}Downloading espeak-ng data...${NC}"
    ESPEAK_URL="${SHERPA_BASE}/tts-models/espeak-ng-data.tar.bz2"
    
    tmp_dir=$(mktemp -d)
    
    curl -L "$ESPEAK_URL" -o "${tmp_dir}/espeak-ng-data.tar.bz2"
    tar -xjf "${tmp_dir}/espeak-ng-data.tar.bz2" -C "${KOKORO_DIR}"
    
    rm -rf "${tmp_dir}"
    
    echo -e "${GREEN}✓ espeak-ng data installed${NC}"
fi

# Verify installation
echo
echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}Verifying installation...${NC}"
echo

MISSING_FILES=()

check_file() {
    if [[ -f "$1" ]]; then
        echo -e "${GREEN}✓ $1${NC}"
    else
        echo -e "${RED}✗ $1 (MISSING)${NC}"
        MISSING_FILES+=("$1")
    fi
}

check_file "${MODELS_DIR}/silero_vad.onnx"
check_file "${MODELS_DIR}/whisper/whisper-small-encoder.int8.onnx"
check_file "${MODELS_DIR}/whisper/whisper-small-decoder.int8.onnx"
check_file "${MODELS_DIR}/whisper/whisper-small-tokens.txt"
check_file "${KOKORO_DIR}/model.onnx"
check_file "${KOKORO_DIR}/voices.bin"
check_file "${KOKORO_DIR}/tokens.txt"

# Check Ollama after model installation
check_ollama

echo
if [[ ${#MISSING_FILES[@]} -eq 0 ]]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}✓ All models installed successfully!${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo
    echo -e "${BLUE}Platform: ${PLATFORM}${NC}"
    echo -e "${BLUE}Recommended hardware acceleration: ${PROVIDER}${NC}"
    echo
    echo "To build and run the voice assistant:"
    echo
    echo "  cd sherpa-go"
    echo "  CGO_ENABLED=1 go build -o voice-assistant ./cmd/assistant"
    echo "  ./voice-assistant --assets ${ASSETS_DIR}"
    echo
    if [[ "$PROVIDER" == "cuda" ]]; then
        echo -e "${YELLOW}CUDA Notes:${NC}"
        echo "  • Ensure CUDA toolkit is installed: sudo apt install nvidia-cuda-toolkit"
        echo "  • Ensure cuDNN is installed for optimal performance"
        echo "  • The application will auto-detect CUDA and use GPU acceleration"
        echo
    elif [[ "$PROVIDER" == "coreml" ]]; then
        echo -e "${YELLOW}CoreML Notes:${NC}"
        echo "  • CoreML will automatically use Apple Neural Engine on M-series chips"
        echo "  • No additional setup required"
        echo
    fi
    echo "Make sure Ollama is running with a model loaded:"
    echo "  ollama run gemma3:1b"
    echo
else
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${RED}✗ Some files are missing. Please check the errors above.${NC}"
    echo -e "${RED}═══════════════════════════════════════════════════════════${NC}"
    exit 1
fi
