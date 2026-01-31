#!/bin/bash
#
# Build script for Voice Assistant (Go + sherpa-onnx)
# Handles platform-specific dependencies and CGO setup
#
# Usage:
#   ./scripts/build.sh          # Build with auto-detected settings
#   ./scripts/build.sh --cuda   # Force CUDA build (requires building sherpa-onnx from source)
#   ./scripts/build.sh --cpu    # Force CPU-only build
#   ./scripts/build.sh --clean  # Clean CUDA build artifacts (~/.voice-assistant/go/lib) and rebuild
#

set -e

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Parse arguments
FORCE_CUDA=false
FORCE_CPU=false
CLEAN_BUILD=false
for arg in "$@"; do
    case $arg in
        --cuda)
            FORCE_CUDA=true
            ;;
        --cpu)
            FORCE_CPU=true
            ;;
        --clean)
            CLEAN_BUILD=true
            ;;
    esac
done

echo -e "${GREEN}Building Voice Assistant (Go + sherpa-onnx)${NC}"
echo

# Ensure CGO is enabled
export CGO_ENABLED=1

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

# =============================================================================
# VERSION SANITY CHECK
# =============================================================================
# Verifies that go.mod sherpa-onnx version matches the SHERPA_VERSION used for
# CUDA builds. This prevents ABI mismatches when the pre-built Go bindings
# expect a different version than the locally compiled CUDA libraries.
#
# This check only runs on Linux (where CUDA builds happen).
# macOS uses pre-built bindings that handle version compatibility internally.
# =============================================================================
verify_go_sherpa_version() {
    local expected_version="$1"

    # Extract version from go.mod
    local go_mod_version
    go_mod_version=$(grep 'k2-fsa/sherpa-onnx-go-linux' "$PROJECT_DIR/go.mod" 2>/dev/null | grep -oE 'v[0-9]+\.[0-9]+\.[0-9]+' | head -1 || echo "")

    if [[ -z "$go_mod_version" ]]; then
        echo -e "${YELLOW}Warning: Could not determine sherpa-onnx version from go.mod${NC}"
        echo -e "${YELLOW}Skipping version check...${NC}"
        return 0
    fi

    if [[ "$go_mod_version" != "$expected_version" ]]; then
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}VERSION MISMATCH DETECTED!${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════════${NC}"
        echo -e "${RED}  go.mod sherpa-onnx-go-linux:  $go_mod_version${NC}"
        echo -e "${RED}  Build script SHERPA_VERSION:  $expected_version${NC}"
        echo
        echo -e "${YELLOW}For CUDA builds, these versions MUST match to avoid ABI mismatches.${NC}"
        echo
        echo -e "${GREEN}To fix this, either:${NC}"
        echo -e "${GREEN}  1. Update go.mod to match the build script:${NC}"
        echo -e "${GREEN}     go get github.com/k2-fsa/sherpa-onnx-go-linux@$expected_version${NC}"
        echo -e "${GREEN}     go get github.com/k2-fsa/sherpa-onnx-go-macos@$expected_version${NC}"
        echo
        echo -e "${GREEN}  2. Or update SHERPA_VERSION in this script to match go.mod:${NC}"
        echo -e "${GREEN}     Edit scripts/build.sh and set SHERPA_VERSION=\"$go_mod_version\"${NC}"
        echo
        echo -e "${YELLOW}See README.md \"Upgrading Dependencies\" for the full upgrade procedure.${NC}"
        echo -e "${RED}═══════════════════════════════════════════════════════════════════════════${NC}"
        exit 1
    fi

    echo -e "${GREEN}Version check passed: go.mod and build script both use $go_mod_version${NC}"
}

echo -e "${YELLOW}Platform: ${OS} ${ARCH}${NC}"

# Function to detect NVIDIA GPU (discrete or Jetson SOC)
detect_nvidia_gpu() {
    # Check for nvidia-smi
    for path in /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi /opt/nvidia/bin/nvidia-smi; do
        if [[ -f "$path" ]]; then
            return 0
        fi
    done

    # Check for discrete GPU device
    if [[ -e /dev/nvidia0 ]]; then
        return 0
    fi

    # Check for Jetson indicators
    for path in /dev/nvhost-gpu /dev/nvhost-ctrl-gpu /dev/nvmap /etc/nv_tegra_release \
                /sys/devices/gpu.0 /sys/devices/17000000.ga10b /sys/devices/17000000.gv11b; do
        if [[ -e "$path" ]]; then
            return 0
        fi
    done

    # Check device tree for tegra/jetson
    if [[ -f /proc/device-tree/compatible ]]; then
        if grep -q "nvidia,tegra\|nvidia,jetson" /proc/device-tree/compatible 2>/dev/null; then
            return 0
        fi
    fi

    return 1
}

# Function to check if CUDA toolkit is available
check_cuda_toolkit() {
    if command -v nvcc &>/dev/null; then
        return 0
    fi
    if [[ -d /usr/local/cuda ]]; then
        return 0
    fi
    # Jetson JetPack includes CUDA
    for cuda_dir in /usr/local/cuda-*; do
        if [[ -d "$cuda_dir" ]]; then
            return 0
        fi
    done
    return 1
}

# Function to get CUDA version (major.minor)
get_cuda_version() {
    local cuda_version=""
    if command -v nvcc &>/dev/null; then
        cuda_version=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    elif [[ -f /usr/local/cuda/version.txt ]]; then
        cuda_version=$(cat /usr/local/cuda/version.txt | grep -oP 'CUDA Version \K[0-9]+\.[0-9]+')
    elif [[ -f /usr/local/cuda/version.json ]]; then
        cuda_version=$(grep -oP '"cuda" *: *\{ *"version" *: *"\K[0-9]+\.[0-9]+' /usr/local/cuda/version.json 2>/dev/null || echo "")
    fi
    # If we got full version, just use it
    if [[ -z "$cuda_version" && -d /usr/local/cuda ]]; then
        # Try to extract from cuda path symlink
        local target=$(readlink -f /usr/local/cuda 2>/dev/null || echo "")
        if [[ "$target" =~ cuda-([0-9]+\.[0-9]+) ]]; then
            cuda_version="${BASH_REMATCH[1]}"
        fi
    fi
    echo "$cuda_version"
}

# Function to get the appropriate ONNX Runtime version for aarch64 GPU based on CUDA version
# See: https://github.com/k2-fsa/sherpa-onnx/blob/main/cmake/onnxruntime-linux-aarch64-gpu.cmake
get_onnxruntime_version_for_cuda() {
    local cuda_ver="$1"
    local cuda_major="${cuda_ver%%.*}"

    case "$cuda_ver" in
        10.2*)
            # Jetson Nano B01
            echo "1.11.0"
            ;;
        11.4*)
            # Jetson Orin NX / JetPack 5.x
            echo "1.16.0"
            ;;
        12.2*)
            # CUDA 12.2 with cudnn8
            echo "1.18.0"
            ;;
        12.6*|12.*)
            # JetPack 6.2+ (CUDA 12.6, cudnn9)
            echo "1.18.1"
            ;;
        11.*)
            # Default for CUDA 11.x - use 1.16.0
            echo "1.16.0"
            ;;
        *)
            # Default to 1.18.1 for unknown CUDA 12+ or newer
            if [[ "$cuda_major" -ge 12 ]]; then
                echo "1.18.1"
            else
                echo "1.16.0"
            fi
            ;;
    esac
}

# Determine if we should build with CUDA
USE_CUDA=false
if [[ "$FORCE_CPU" == "true" ]]; then
    echo -e "${YELLOW}CPU-only build forced via --cpu flag${NC}"
    USE_CUDA=false
elif [[ "$FORCE_CUDA" == "true" ]]; then
    echo -e "${YELLOW}CUDA build forced via --cuda flag${NC}"
    USE_CUDA=true
elif [[ "$OS" == "Linux" ]] && detect_nvidia_gpu; then
    echo -e "${YELLOW}NVIDIA GPU detected${NC}"
    if check_cuda_toolkit; then
        echo -e "${GREEN}CUDA toolkit available - enabling GPU support${NC}"
        USE_CUDA=true
    else
        echo -e "${YELLOW}CUDA toolkit not found - using CPU-only build${NC}"
        echo -e "${YELLOW}To enable GPU support, install CUDA toolkit or use JetPack${NC}"
        USE_CUDA=false
    fi
fi

case "$OS" in
    Darwin)
        echo -e "${YELLOW}macOS detected${NC}"
        echo -e "${GREEN}  ℹ️  Version checks skipped: macOS uses pre-built sherpa-onnx-go-macos bindings${NC}"
        echo -e "${GREEN}     that handle version compatibility internally.${NC}"
        ;;
    Linux)
        echo -e "${YELLOW}Linux detected${NC}"
        # Check for ALSA dev libraries
        if ! pkg-config --exists alsa 2>/dev/null; then
            echo -e "${RED}Warning: ALSA development libraries not found.${NC}"
            echo "Install with: sudo apt-get install libasound2-dev"
        fi
        ;;
    MINGW*|MSYS*|CYGWIN*)
        echo -e "${YELLOW}Windows detected${NC}"
        ;;
    *)
        echo -e "${RED}Unknown OS: $OS${NC}"
        ;;
esac

echo

# Build sherpa-onnx from source with CUDA if needed
SHERPA_ONNX_LIB_DIR=""
if [[ "$USE_CUDA" == "true" && "$OS" == "Linux" ]]; then
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}Building sherpa-onnx with CUDA support...${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo

    SHERPA_BUILD_DIR="$PROJECT_DIR/.sherpa-onnx-build"
    # Runtime libraries go to ~/.voice-assistant/go/lib for portability
    # Note: Using separate directory from Rust impl to avoid version conflicts
    SHERPA_INSTALL_DIR="$HOME/.voice-assistant/go"
    SHERPA_GO_LOCAL="$PROJECT_DIR/.sherpa-onnx-go-local"
    # IMPORTANT: This version must match the sherpa-onnx-go-linux/macos versions in go.mod
    # The sanity check below will fail the build if they drift apart.
    # See README.md "Upgrading Dependencies" for the upgrade procedure.
    SHERPA_VERSION="v1.12.22"
    BUILD_MARKER="$SHERPA_INSTALL_DIR/.build-complete-${SHERPA_VERSION}"

    # Verify go.mod version matches before building
    verify_go_sherpa_version "$SHERPA_VERSION"
    echo

    # Clean build if requested
    if [[ "$CLEAN_BUILD" == "true" ]]; then
        echo -e "${YELLOW}Cleaning previous CUDA build...${NC}"
        rm -rf "$SHERPA_BUILD_DIR" "$SHERPA_INSTALL_DIR/lib" "$SHERPA_GO_LOCAL"
        rm -f "$SHERPA_INSTALL_DIR/.build-complete-"*
    fi

    # Check if we already have a complete CUDA build for this version
    # We use a marker file to ensure the build completed successfully
    if [[ -f "$BUILD_MARKER" ]] && \
       ls "$SHERPA_INSTALL_DIR/lib/"libsherpa-onnx-c-api.so* &>/dev/null && \
       [[ -d "$SHERPA_GO_LOCAL" ]] && \
       ls "$SHERPA_GO_LOCAL/lib/"*/libsherpa-onnx-c-api.so* &>/dev/null; then
        echo -e "${GREEN}Using existing sherpa-onnx CUDA build (${SHERPA_VERSION})${NC}"
        echo -e "${GREEN}  Libraries: $SHERPA_INSTALL_DIR/lib${NC}"
        echo -e "${GREEN}  Go bindings: $SHERPA_GO_LOCAL${NC}"
        echo -e "${YELLOW}  (Use --clean to force rebuild)${NC}"
    else
        # Install build dependencies
        echo -e "${YELLOW}Checking build dependencies...${NC}"
        if ! command -v cmake &>/dev/null; then
            echo -e "${RED}CMake is required but not installed.${NC}"
            echo "Install with: sudo apt-get install cmake"
            exit 1
        fi

        # Clone sherpa-onnx if needed
        if [[ ! -d "$SHERPA_BUILD_DIR" ]]; then
            echo -e "${YELLOW}Cloning sherpa-onnx...${NC}"
            git clone --depth 1 --branch "$SHERPA_VERSION" https://github.com/k2-fsa/sherpa-onnx.git "$SHERPA_BUILD_DIR"
        else
            echo -e "${YELLOW}Using existing sherpa-onnx source...${NC}"
        fi

        # Set CUDA path
        if [[ -z "$CUDA_HOME" ]]; then
            if [[ -d /usr/local/cuda ]]; then
                export CUDA_HOME=/usr/local/cuda
            else
                # Find any CUDA version
                for cuda_dir in /usr/local/cuda-*; do
                    if [[ -d "$cuda_dir" ]]; then
                        export CUDA_HOME="$cuda_dir"
                        break
                    fi
                done
            fi
        fi

        if [[ -n "$CUDA_HOME" ]]; then
            echo -e "${GREEN}Using CUDA from: $CUDA_HOME${NC}"
            export PATH="$CUDA_HOME/bin:$PATH"
            export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"

            # Detect CUDA version for compatibility info
            CUDA_VERSION=$(get_cuda_version)
            if [[ -n "$CUDA_VERSION" ]]; then
                echo -e "${GREEN}CUDA version: $CUDA_VERSION${NC}"
                CUDA_MAJOR="${CUDA_VERSION%%.*}"
                if [[ "$CUDA_MAJOR" -lt 10 ]]; then
                    echo -e "${RED}Warning: CUDA $CUDA_VERSION detected. sherpa-onnx requires CUDA 10.2 or higher.${NC}"
                    echo -e "${RED}GPU acceleration may not work. Consider upgrading JetPack.${NC}"
                fi

                # Get the appropriate ONNX Runtime version for this CUDA version
                ONNX_RT_VERSION=$(get_onnxruntime_version_for_cuda "$CUDA_VERSION")
                echo -e "${GREEN}Using ONNX Runtime version: $ONNX_RT_VERSION (for CUDA $CUDA_VERSION)${NC}"
            else
                # Default ONNX Runtime version if we can't detect CUDA version
                ONNX_RT_VERSION="1.18.1"
                echo -e "${YELLOW}Could not detect CUDA version, using default ONNX Runtime: $ONNX_RT_VERSION${NC}"
            fi
        else
            ONNX_RT_VERSION="1.18.1"
            echo -e "${YELLOW}CUDA_HOME not set, using default ONNX Runtime: $ONNX_RT_VERSION${NC}"
        fi

        # Build sherpa-onnx with CUDA
        echo -e "${YELLOW}Building sherpa-onnx (this may take 10-20 minutes)...${NC}"
        mkdir -p "$SHERPA_BUILD_DIR/build-cuda"
        cd "$SHERPA_BUILD_DIR/build-cuda"

        cmake .. \
            -DCMAKE_BUILD_TYPE=Release \
            -DCMAKE_INSTALL_PREFIX="$SHERPA_INSTALL_DIR" \
            -DBUILD_SHARED_LIBS=ON \
            -DSHERPA_ONNX_ENABLE_GPU=ON \
            -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION="$ONNX_RT_VERSION" \
            -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
            -DSHERPA_ONNX_ENABLE_BINARY=OFF \
            -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
            -DSHERPA_ONNX_ENABLE_C_API=ON

        cmake --build . --config Release -j$(nproc)
        cmake --install .

        # Copy the C API header to include directory
        mkdir -p "$SHERPA_INSTALL_DIR/include"
        cp -f "$SHERPA_BUILD_DIR/sherpa-onnx/c-api/c-api.h" "$SHERPA_INSTALL_DIR/include/"

        cd "$PROJECT_DIR"
        echo -e "${GREEN}sherpa-onnx C library build complete${NC}"

        # Create a local Go package that wraps our CUDA-enabled libraries
        echo -e "${YELLOW}Creating local Go bindings package...${NC}"
        mkdir -p "$SHERPA_GO_LOCAL/lib/${ARCH}-unknown-linux-gnu"

        # Copy the built libraries
        cp -a "$SHERPA_INSTALL_DIR/lib/"*.so* "$SHERPA_GO_LOCAL/lib/${ARCH}-unknown-linux-gnu/"

        # Get the original package location
        ORIG_PKG=$(go list -m -f '{{.Dir}}' github.com/k2-fsa/sherpa-onnx-go-linux 2>/dev/null || echo "")
        if [[ -z "$ORIG_PKG" || ! -d "$ORIG_PKG" ]]; then
            echo -e "${YELLOW}Downloading original Go package...${NC}"
            go mod download github.com/k2-fsa/sherpa-onnx-go-linux@${SHERPA_VERSION}
            ORIG_PKG=$(go list -m -f '{{.Dir}}' github.com/k2-fsa/sherpa-onnx-go-linux)
        fi

        # Copy Go source files and headers
        cp "$ORIG_PKG/"*.go "$SHERPA_GO_LOCAL/" 2>/dev/null || true
        cp "$ORIG_PKG/c-api.h" "$SHERPA_GO_LOCAL/" 2>/dev/null || true

        # Also copy the C API header from our build (in case it's newer)
        cp -f "$SHERPA_BUILD_DIR/sherpa-onnx/c-api/c-api.h" "$SHERPA_GO_LOCAL/"

        # Create go.mod for the local package
        cat > "$SHERPA_GO_LOCAL/go.mod" << EOF
module github.com/k2-fsa/sherpa-onnx-go-linux

go 1.25
EOF

        echo -e "${GREEN}Local Go bindings package created${NC}"

        # Create build marker to indicate successful completion
        touch "$BUILD_MARKER"
        echo -e "${GREEN}Build marker created: $BUILD_MARKER${NC}"
    fi

    SHERPA_ONNX_LIB_DIR="$SHERPA_INSTALL_DIR/lib"

    # Create a run wrapper script that sets up the environment (only if it doesn't exist)
    if [[ ! -f "$PROJECT_DIR/run-voice-assistant.sh" ]]; then
        echo -e "${YELLOW}Creating run wrapper script...${NC}"
        cat > "$PROJECT_DIR/run-voice-assistant.sh" << 'WRAPPER'
#!/bin/bash
# Wrapper script to run voice-assistant with proper CUDA library paths

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Find CUDA home
CUDA_HOME="${CUDA_HOME:-}"
if [[ -z "$CUDA_HOME" ]]; then
    if [[ -d /usr/local/cuda ]]; then
        CUDA_HOME=/usr/local/cuda
    else
        for cuda_dir in /usr/local/cuda-*; do
            if [[ -d "$cuda_dir" ]]; then
                CUDA_HOME="$cuda_dir"
                break
            fi
        done
    fi
fi

# Set up library paths - check ~/.voice-assistant/go/lib for portable deployment
if [[ -d "$HOME/.voice-assistant/go/lib" ]]; then
    export LD_LIBRARY_PATH="$HOME/.voice-assistant/go/lib:${LD_LIBRARY_PATH:-}"
fi

if [[ -n "$CUDA_HOME" && -d "$CUDA_HOME/lib64" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"
fi

# Also check for Jetson-specific paths
if [[ -d /usr/lib/aarch64-linux-gnu/tegra ]]; then
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH"
fi

# Run the assistant
exec "$SCRIPT_DIR/voice-assistant" "$@"
WRAPPER
        chmod +x "$PROJECT_DIR/run-voice-assistant.sh"
    else
        echo -e "${GREEN}Run wrapper script already exists${NC}"
        # Ensure it's executable even if it already exists
        chmod +x "$PROJECT_DIR/run-voice-assistant.sh"
    fi

    # Add replace directive to go.mod if not present
    if ! grep -q "replace github.com/k2-fsa/sherpa-onnx-go-linux" go.mod; then
        echo -e "${YELLOW}Adding go.mod replace directive for CUDA bindings...${NC}"
        echo "" >> go.mod
        echo "replace github.com/k2-fsa/sherpa-onnx-go-linux => ./.sherpa-onnx-go-local" >> go.mod
    fi

    echo -e "${GREEN}Using CUDA-enabled sherpa-onnx from: $SHERPA_ONNX_LIB_DIR${NC}"
    echo -e "${GREEN}Runtime libraries installed to: ~/.voice-assistant/go/lib${NC}"
    echo -e "${GREEN}For portable deployment, copy ~/.voice-assistant/go and the binary to the target machine.${NC}"
    echo
else
    # Remove replace directive if present (for CPU builds)
    if grep -q "replace github.com/k2-fsa/sherpa-onnx-go-linux" go.mod 2>/dev/null; then
        echo -e "${YELLOW}Removing CUDA replace directive from go.mod...${NC}"
        if [[ "$OS" == "Darwin" ]]; then
            sed -i '' '/replace github.com\/k2-fsa\/sherpa-onnx-go-linux/d' go.mod
        else
            sed -i '/replace github.com\/k2-fsa\/sherpa-onnx-go-linux/d' go.mod
        fi
    fi
fi

# Download dependencies
echo -e "${GREEN}Downloading dependencies...${NC}"
go mod download

echo
echo -e "${GREEN}Running go mod tidy...${NC}"
go mod tidy

echo
echo -e "${GREEN}Building...${NC}"

# Set CGO optimization flags for edge devices (ARM SIMD, etc.)
# Enable aggressive optimizations in sherpa-onnx C++ code
export CGO_CFLAGS="${CGO_CFLAGS:-} -O3"
export CGO_CXXFLAGS="${CGO_CXXFLAGS:-} -O3"

# For ARM platforms (Jetson), enable NEON SIMD optimizations
if [[ "$ARCH" == "aarch64" || "$ARCH" == "arm64" ]]; then
    echo -e "${GREEN}Enabling ARM NEON SIMD optimizations for edge devices...${NC}"
    export CGO_CFLAGS="${CGO_CFLAGS} -march=native -mtune=native"
    export CGO_CXXFLAGS="${CGO_CXXFLAGS} -march=native -mtune=native"
fi

# Build (the replace directive in go.mod handles CUDA vs CPU)
go build -ldflags="-s -w" -o voice-assistant ./cmd/assistant

if [[ -f "voice-assistant" ]]; then
    echo
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    if [[ "$USE_CUDA" == "true" ]]; then
        echo -e "${GREEN}✓ Build successful (CUDA enabled): ./voice-assistant${NC}"
    else
        echo -e "${GREEN}✓ Build successful (CPU only): ./voice-assistant${NC}"
    fi
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo
    echo "Before running, ensure you have:"
    echo "  1. Downloaded models: ./scripts/setup.sh"
    echo "  2. Started Ollama: ollama serve"
    echo "  3. Loaded a model: ollama run gemma3:1b"
    echo
    if [[ "$USE_CUDA" == "true" && -n "$SHERPA_ONNX_LIB_DIR" ]]; then
        echo -e "${YELLOW}CUDA build notes:${NC}"
        echo "  • A wrapper script has been created: ./run-voice-assistant.sh"
        echo "  • The wrapper sets up CUDA library paths automatically"
        echo "  • Libraries are in: $SHERPA_ONNX_LIB_DIR"
        if [[ -n "$CUDA_VERSION" ]]; then
            echo "  • Built for CUDA: $CUDA_VERSION (ONNX Runtime: $ONNX_RT_VERSION)"
        fi
        echo
        echo -e "${YELLOW}Supported Jetson configurations:${NC}"
        echo "  • Jetson Nano B01:  CUDA 10.2 → ONNX Runtime 1.11.0"
        echo "  • Jetson Orin NX:   CUDA 11.4 → ONNX Runtime 1.16.0"
        echo "  • JetPack 6.x:      CUDA 12.2+→ ONNX Runtime 1.18.0/1.18.1"
        echo
        echo -e "${YELLOW}If you see library errors at runtime:${NC}"
        echo "  1. Make sure CUDA version matches your JetPack"
        echo "  2. Use the wrapper script: ./run-voice-assistant.sh"
        echo "  3. Or set LD_LIBRARY_PATH manually:"
        echo "     export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$SHERPA_ONNX_LIB_DIR:\$LD_LIBRARY_PATH"
        echo
        echo "Run with: ./run-voice-assistant.sh"
        echo "Or directly: ./voice-assistant (may need LD_LIBRARY_PATH set)"
    else
        echo "Then run: ./voice-assistant"
    fi
else
    echo -e "${RED}Build failed!${NC}"
    exit 1
fi
