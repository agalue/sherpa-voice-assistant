#!/bin/bash
# Build script for the Voice Assistant
# Supports debug and release builds with automatic CUDA detection

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

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

print_usage() {
    echo "Usage: $0 [OPTIONS]"
    echo ""
    echo "Build the Voice Assistant application."
    echo ""
    echo "Options:"
    echo "  -r, --release    Build in release mode (optimized)"
    echo "  -d, --debug      Build in debug mode (default)"
    echo "  -c, --clean      Clean build artifacts before building"
    echo "  -t, --test       Run tests after building"
    echo "  --cuda           Force CUDA build (Linux only, builds sherpa-onnx from source)"
    echo "  --cpu            Force CPU-only build (disable CUDA)"
    echo "  -h, --help       Show this help message"
}

# Default options
BUILD_MODE="release"
CLEAN=false
RUN_TESTS=false
FORCE_CUDA=false
FORCE_CPU=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -r|--release)
            BUILD_MODE="release"
            shift
            ;;
        -d|--debug)
            BUILD_MODE="debug"
            shift
            ;;
        -c|--clean)
            CLEAN=true
            shift
            ;;
        -t|--test)
            RUN_TESTS=true
            shift
            ;;
        --cuda)
            FORCE_CUDA=true
            shift
            ;;
        --cpu)
            FORCE_CPU=true
            shift
            ;;
        -h|--help)
            print_usage
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            print_usage
            exit 1
            ;;
    esac
done

cd "${PROJECT_DIR}"

# Check for Rust toolchain
if ! command -v cargo &> /dev/null; then
    log_error "Cargo (Rust) is not installed"
    log_error "Install from: https://rustup.rs"
    exit 1
fi

# Detect platform
OS=$(uname -s)
ARCH=$(uname -m)

log_info "Platform: ${OS} ${ARCH}"

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

# Function to get CUDA version
get_cuda_version() {
    local cuda_version=""
    if command -v nvcc &>/dev/null; then
        cuda_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    elif [[ -f /usr/local/cuda/version.txt ]]; then
        cuda_version=$(cat /usr/local/cuda/version.txt | grep -oP 'CUDA Version \K[0-9]+\.[0-9]+')
    elif [[ -f /usr/local/cuda/version.json ]]; then
        cuda_version=$(grep -oP '"cuda" *: *\{ *"version" *: *"\K[0-9]+\.[0-9]+' /usr/local/cuda/version.json 2>/dev/null || echo "")
    fi
    echo "$cuda_version"
}

# Function to get the appropriate ONNX Runtime version for aarch64 GPU based on CUDA version
# See: https://github.com/k2-fsa/sherpa-onnx/releases for available GPU builds
get_onnxruntime_version_for_cuda() {
    local cuda_ver="$1"
    local cuda_major="${cuda_ver%%.*}"

    case "$cuda_ver" in
        10.2*)
            # Jetson Nano B01 / JetPack 4.x
            echo "1.11.0"
            ;;
        11.4*)
            # Jetson Orin / JetPack 5.x
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

# =============================================================================
# VERSION SANITY CHECK
# =============================================================================
# Verifies that the sherpa-onnx source version matches the C API headers bundled
# in sherpa-rs-sys. This prevents ABI mismatches when linking against locally
# compiled CUDA libraries.
#
# This check only runs on Linux (where CUDA builds happen).
# macOS uses pre-built static libraries that handle compatibility internally.
# =============================================================================

# Known version mappings: sherpa-rs-sys version -> expected sherpa-onnx version
# Update this function when upgrading sherpa-rs
# Note: Using a function instead of associative array for bash 3.2 compatibility (macOS)
get_sherpa_onnx_version_for_rs_sys() {
    local rs_sys_version="$1"
    case "$rs_sys_version" in
        0.6.8) echo "v1.12.10" ;;
        0.6.7) echo "v1.12.10" ;;
        0.6.6) echo "v1.12.10" ;;
        *) echo "" ;;
    esac
}

verify_rust_sherpa_version() {
    local expected_version="$1"

    # Check if Cargo.lock exists
    if [[ ! -f "$PROJECT_DIR/Cargo.lock" ]]; then
        echo -e "${YELLOW}[WARN]${NC} Cargo.lock not found, skipping version check" >&2
        echo -e "${YELLOW}[WARN]${NC} Run 'cargo build' first to generate Cargo.lock" >&2
        return 0
    fi

    # Extract sherpa-rs-sys version from Cargo.lock
    local sherpa_rs_sys_version
    sherpa_rs_sys_version=$(grep -A1 'name = "sherpa-rs-sys"' "$PROJECT_DIR/Cargo.lock" | grep 'version' | head -1 | grep -oE '[0-9]+\.[0-9]+\.[0-9]+' || echo "")

    if [[ -z "$sherpa_rs_sys_version" ]]; then
        echo -e "${YELLOW}[WARN]${NC} Could not determine sherpa-rs-sys version from Cargo.lock" >&2
        return 0
    fi

    # Look up expected sherpa-onnx version for this sherpa-rs-sys version
    local mapped_version
    mapped_version=$(get_sherpa_onnx_version_for_rs_sys "$sherpa_rs_sys_version")

    if [[ -z "$mapped_version" ]]; then
        echo -e "${YELLOW}[WARN]${NC} Unknown sherpa-rs-sys version: $sherpa_rs_sys_version" >&2
        echo -e "${YELLOW}[WARN]${NC} Add mapping to get_sherpa_onnx_version_for_rs_sys() in scripts/build.sh" >&2
        echo -e "${YELLOW}[WARN]${NC} Proceeding with version $expected_version (may cause ABI mismatch)" >&2
        return 0
    fi

    if [[ "$mapped_version" != "$expected_version" ]]; then
        echo -e "${RED}[ERROR]${NC} ═══════════════════════════════════════════════════════════════════════════" >&2
        echo -e "${RED}[ERROR]${NC} VERSION MISMATCH DETECTED!" >&2
        echo -e "${RED}[ERROR]${NC} ═══════════════════════════════════════════════════════════════════════════" >&2
        echo -e "${RED}[ERROR]${NC}   sherpa-rs-sys version (Cargo.lock):  $sherpa_rs_sys_version" >&2
        echo -e "${RED}[ERROR]${NC}   Expected sherpa-onnx version:        $mapped_version" >&2
        echo -e "${RED}[ERROR]${NC}   Build script sherpa_version:         $expected_version" >&2
        echo -e "${RED}[ERROR]${NC}" >&2
        echo -e "${YELLOW}[INFO]${NC} For CUDA builds, the sherpa-onnx source version must match the" >&2
        echo -e "${YELLOW}[INFO]${NC} C API headers bundled in sherpa-rs-sys to avoid ABI mismatches." >&2
        echo -e "${RED}[ERROR]${NC}" >&2
        echo -e "${GREEN}[FIX]${NC}  Update sherpa_version in build_sherpa_onnx() to: $mapped_version" >&2
        echo -e "${RED}[ERROR]${NC}" >&2
        echo -e "${YELLOW}[INFO]${NC} See README.md \"Upgrading Dependencies\" for the full upgrade procedure." >&2
        echo -e "${RED}[ERROR]${NC} ═══════════════════════════════════════════════════════════════════════════" >&2
        return 1
    fi

    echo -e "${GREEN}[INFO]${NC} Version check passed: sherpa-rs-sys $sherpa_rs_sys_version → sherpa-onnx $expected_version" >&2
    return 0
}

# Function to build sherpa-onnx from source with CUDA support
# This ensures ABI compatibility between the C API headers and the built libraries
# Note: All log output goes to stderr so stdout only contains the path
build_sherpa_onnx() {
    local cuda_version="$1"
    local onnx_version=$(get_onnxruntime_version_for_cuda "$cuda_version")

    # =============================================================================
    # IMPORTANT: sherpa-onnx version for CUDA builds
    # =============================================================================
    # This version MUST match the C API headers bundled in sherpa-rs-sys.
    # sherpa-rs-sys bundles headers from a specific sherpa-onnx version, and using
    # a different version causes ABI mismatches (e.g., struct field count differences).
    #
    # Known version mappings (update this table when upgrading sherpa-rs):
    #   sherpa-rs 0.6.x -> sherpa-rs-sys 0.6.8 -> sherpa-onnx v1.12.10 headers
    #
    # To determine the correct version when upgrading sherpa-rs:
    #   1. Check Cargo.lock for the resolved sherpa-rs-sys version
    #   2. Look at sherpa-rs-sys source: https://github.com/thewh1teagle/sherpa-rs
    #   3. Find the bundled c-api.h and compare with sherpa-onnx releases
    #   4. Note: dist.json may claim a different version than actual bundled headers!
    #   5. Test CUDA builds and watch for struct field mismatch errors
    #
    # Example ABI mismatch error: v1.12.15 added wenet_ctc field to
    # SherpaOnnxOfflineModelConfig that isn't in the v1.12.10 bundled headers.
    # =============================================================================
    local sherpa_version="v1.12.10"

    # Verify version compatibility before building
    if ! verify_rust_sherpa_version "$sherpa_version"; then
        return 1
    fi

    local build_dir="${PROJECT_DIR}/.sherpa-onnx-build"
    # Runtime libraries go to ~/.voice-assistant/rust for portability
    # Note: Using separate directory from Go impl to avoid version conflicts
    local install_dir="$HOME/.voice-assistant/rust"
    local build_marker="${install_dir}/.build-complete-${sherpa_version}-cuda${cuda_version}"

    echo -e "${GREEN}[INFO]${NC} CUDA version: $cuda_version -> ONNX Runtime: $onnx_version" >&2
    echo -e "${GREEN}[INFO]${NC} sherpa-onnx version: $sherpa_version" >&2

    # Check if we already have a complete build for this version
    if [[ -f "$build_marker" ]] && ls "$install_dir/lib/"libsherpa-onnx-c-api.so* &>/dev/null 2>&1; then
        echo -e "${GREEN}[INFO]${NC} Using existing sherpa-onnx CUDA build" >&2
        echo -e "${GREEN}[INFO]${NC}   Libraries: $install_dir/lib" >&2
        echo -e "${YELLOW}[WARN]${NC}   (Use --clean to force rebuild)" >&2
        echo "$install_dir"
        return 0
    fi

    # Check build dependencies
    if ! command -v cmake &>/dev/null; then
        echo -e "${RED}[ERROR]${NC} CMake is required but not installed." >&2
        echo -e "${RED}[ERROR]${NC} Install with: sudo apt-get install cmake" >&2
        return 1
    fi

    if ! command -v git &>/dev/null; then
        echo -e "${RED}[ERROR]${NC} Git is required but not installed." >&2
        echo -e "${RED}[ERROR]${NC} Install with: sudo apt-get install git" >&2
        return 1
    fi

    # Clone sherpa-onnx if needed
    if [[ ! -d "$build_dir" ]]; then
        echo -e "${GREEN}[INFO]${NC} Cloning sherpa-onnx ${sherpa_version}..." >&2
        git clone --depth 1 --branch "$sherpa_version" https://github.com/k2-fsa/sherpa-onnx.git "$build_dir" >&2
    else
        echo -e "${GREEN}[INFO]${NC} Using existing sherpa-onnx source in $build_dir" >&2
    fi

    # Build sherpa-onnx with CUDA
    echo -e "${GREEN}[INFO]${NC} Building sherpa-onnx with CUDA support (this may take 10-20 minutes)..." >&2
    mkdir -p "$build_dir/build-cuda"
    cd "$build_dir/build-cuda"

    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DCMAKE_INSTALL_PREFIX="$install_dir" \
        -DBUILD_SHARED_LIBS=ON \
        -DSHERPA_ONNX_ENABLE_GPU=ON \
        -DSHERPA_ONNX_LINUX_ARM64_GPU_ONNXRUNTIME_VERSION="$onnx_version" \
        -DSHERPA_ONNX_ENABLE_WEBSOCKET=OFF \
        -DSHERPA_ONNX_ENABLE_BINARY=OFF \
        -DSHERPA_ONNX_ENABLE_PORTAUDIO=OFF \
        -DSHERPA_ONNX_ENABLE_C_API=ON >&2

    if ! cmake --build . --config Release -j$(nproc) >&2; then
        echo -e "${RED}[ERROR]${NC} CMake build failed" >&2
        cd "$PROJECT_DIR"
        return 1
    fi

    if ! cmake --install . >&2; then
        echo -e "${RED}[ERROR]${NC} CMake install failed" >&2
        cd "$PROJECT_DIR"
        return 1
    fi

    # Copy the C API header to include directory
    mkdir -p "$install_dir/include"
    cp -f "$build_dir/sherpa-onnx/c-api/c-api.h" "$install_dir/include/"

    cd "$PROJECT_DIR"

    # Verify the build succeeded
    if ! ls "$install_dir/lib/"libsherpa-onnx-c-api.so* &>/dev/null 2>&1; then
        echo -e "${RED}[ERROR]${NC} Build completed but libraries not found in $install_dir/lib" >&2
        return 1
    fi

    # Create build marker
    touch "$build_marker"

    echo -e "${GREEN}[INFO]${NC} sherpa-onnx build complete!" >&2
    echo -e "${GREEN}[INFO]${NC} Libraries installed to: $install_dir/lib" >&2

    echo "$install_dir"
    return 0
}

# Setup CUDA environment if available
setup_cuda_env() {
    if [[ -z "$CUDA_HOME" ]]; then
        if [[ -d /usr/local/cuda ]]; then
            export CUDA_HOME=/usr/local/cuda
        else
            for cuda_dir in /usr/local/cuda-*; do
                if [[ -d "$cuda_dir" ]]; then
                    export CUDA_HOME="$cuda_dir"
                    break
                fi
            done
        fi
    fi

    if [[ -n "$CUDA_HOME" ]]; then
        export PATH="$CUDA_HOME/bin:$PATH"
        export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
        log_info "CUDA_HOME: $CUDA_HOME"
    fi
}

# Determine if we should build with CUDA
USE_CUDA=false
if [[ "$OS" == "Linux" ]]; then
    if [[ "$FORCE_CPU" == "true" ]]; then
        log_info "CPU-only build forced via --cpu flag"
        USE_CUDA=false
    elif [[ "$FORCE_CUDA" == "true" ]]; then
        log_info "CUDA build forced via --cuda flag"
        USE_CUDA=true
    elif detect_nvidia_gpu; then
        log_info "NVIDIA GPU detected"
        if check_cuda_toolkit; then
            log_info "CUDA toolkit available - enabling GPU support"
            USE_CUDA=true
        else
            log_warn "CUDA toolkit not found - using CPU-only build"
            log_warn "To enable GPU support, install CUDA toolkit or use JetPack"
            USE_CUDA=false
        fi
    fi

    if [[ "$USE_CUDA" == "true" ]]; then
        setup_cuda_env
        CUDA_VERSION=$(get_cuda_version)
        if [[ -n "$CUDA_VERSION" ]]; then
            log_info "CUDA version: $CUDA_VERSION"
        fi
        # sherpa-rs uses dynamic linking for CUDA
        export SHERPA_BUILD_SHARED_LIBS=1

        # For Jetson (aarch64), build sherpa-onnx from source with correct CUDA version
        if [[ "$ARCH" == "aarch64" ]]; then
            if [[ -z "$CUDA_VERSION" ]]; then
                log_error "Could not detect CUDA version"
                log_error "Please ensure CUDA toolkit is properly installed"
                exit 1
            fi

            echo ""
            echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
            echo -e "${GREEN}Building sherpa-onnx from source with CUDA $CUDA_VERSION${NC}"
            echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
            echo ""

            # Clean sherpa build if requested
            if [[ "$CLEAN" == "true" ]]; then
                log_info "Cleaning previous sherpa-onnx build..."
                rm -rf "${PROJECT_DIR}/.sherpa-onnx-build" "$HOME/.voice-assistant/rust/lib"
                rm -f "$HOME/.voice-assistant/rust/.build-complete-"*
            fi

            SHERPA_INSTALL_DIR=$(build_sherpa_onnx "$CUDA_VERSION")
            if [[ $? -eq 0 && -n "$SHERPA_INSTALL_DIR" && -d "$SHERPA_INSTALL_DIR" ]]; then
                # Set SHERPA_LIB_PATH to tell sherpa-rs to use our built libraries
                export SHERPA_LIB_PATH="${SHERPA_INSTALL_DIR}"
                log_info "Using built sherpa-onnx libraries from: $SHERPA_LIB_PATH"
            else
                log_error "Failed to build sherpa-onnx from source"
                exit 1
            fi
        fi
    fi
elif [[ "$OS" == "Darwin" ]]; then
    log_info "macOS detected - using CoreML for acceleration"
    log_info "  ℹ️  Version checks skipped: macOS uses pre-built static sherpa-rs bindings"
    log_info "     that handle version compatibility internally."
fi

# Check platform-specific dependencies
case "$OS" in
    Linux)
        if ! pkg-config --exists alsa 2>/dev/null; then
            log_warn "ALSA development libraries not found"
            log_warn "Install with: sudo apt-get install libasound2-dev"
        fi
        ;;
esac

# Clean if requested
if [ "$CLEAN" = true ]; then
    log_info "Cleaning build artifacts..."
    cargo clean
fi

# Build
log_info "Building in ${BUILD_MODE} mode..."

CARGO_ARGS=""
if [ "$BUILD_MODE" = "release" ]; then
    CARGO_ARGS="--release"
    BINARY_PATH="target/release/voice-assistant"
else
    BINARY_PATH="target/debug/voice-assistant"
fi

# Set platform-specific build flags
if [[ "$OS" == "Linux" && "$USE_CUDA" != "true" ]]; then
    # Static linking requires this flag on Linux
    export RUSTFLAGS="${RUSTFLAGS:-} -C relocation-model=dynamic-no-pic"
elif [[ "$OS" == "Darwin" ]]; then
    # On macOS, we use static linking via Cargo.toml, but need to ensure
    # proper framework linking for CoreML and other system libraries
    log_info "Building with static sherpa-onnx linking..."
fi

# Enable SIMD optimizations for target CPU
# For Jetson Orin: Enables ARM Cortex-A78AE NEON instructions
# For x86_64: Enables AVX/AVX2/FMA instructions available on host
if [[ "$BUILD_MODE" == "release" ]]; then
    log_info "Enabling SIMD optimizations for target CPU architecture..."
    export RUSTFLAGS="${RUSTFLAGS:-} -C target-cpu=native"
fi

# For Linux CUDA builds, set rpath to find libraries relative to the binary
# This embeds the library search path directly into the binary
if [[ "$OS" == "Linux" && "$USE_CUDA" == "true" ]]; then
    log_info "Setting rpath for CUDA libraries..."
    # Use $ORIGIN to make the path relative to the binary location
    export RUSTFLAGS="${RUSTFLAGS:-} -C link-arg=-Wl,-rpath,\$ORIGIN"
fi

cargo build $CARGO_ARGS

# Run tests if requested
if [ "$RUN_TESTS" = true ]; then
    log_info "Running tests..."
    cargo test $CARGO_ARGS
fi

log_info "Build complete!"
echo ""
echo "Binary location: ${PROJECT_DIR}/${BINARY_PATH}"
echo ""

# Post-build: Fix macOS dynamic library paths if needed (for non-static builds)
if [[ "$OS" == "Darwin" ]]; then
    # Check if the binary has unresolved dylib references
    if otool -L "${BINARY_PATH}" 2>/dev/null | grep -q "@rpath.*libonnxruntime"; then
        log_warn "Binary has dynamic library dependencies that may need fixing..."

        # Find where the libraries were downloaded by sherpa-rs
        SHERPA_LIB_DIR=""
        for dir in "${HOME}/.cache/sherpa-rs" "${PROJECT_DIR}/target/${BUILD_MODE}/build"/sherpa-*/out/lib; do
            if [[ -d "$dir" ]] && ls "$dir"/*.dylib &>/dev/null 2>&1; then
                SHERPA_LIB_DIR="$dir"
                break
            fi
        done

        if [[ -n "$SHERPA_LIB_DIR" ]]; then
            log_info "Found sherpa libraries in: $SHERPA_LIB_DIR"

            # Create a run wrapper that sets DYLD_LIBRARY_PATH
            RUN_SCRIPT="${PROJECT_DIR}/run-voice-assistant.sh"
            cat > "$RUN_SCRIPT" << EOF
#!/bin/bash
# Wrapper script to run voice-assistant with proper library paths on macOS

SCRIPT_DIR="\$(cd "\$(dirname "\${BASH_SOURCE[0]}")" && pwd)"
export DYLD_LIBRARY_PATH="${SHERPA_LIB_DIR}:\${DYLD_LIBRARY_PATH:-}"
exec "\$SCRIPT_DIR/${BINARY_PATH}" "\$@"
EOF
            chmod +x "$RUN_SCRIPT"
            log_info "Created run wrapper script: ${RUN_SCRIPT}"
            echo ""
            echo "To run the assistant:"
            echo "  ./run-voice-assistant.sh [options]"
        else
            log_warn "Could not find sherpa-onnx libraries."
            log_warn "If you see dyld errors, try rebuilding with: cargo clean && cargo build --release"
        fi
    else
        echo "To run the assistant:"
        if [ "$BUILD_MODE" = "release" ]; then
            echo "  ./target/release/voice-assistant"
        else
            echo "  ./target/debug/voice-assistant"
        fi
    fi
# Create a run script for CUDA builds that sets up LD_LIBRARY_PATH
elif [[ "$USE_CUDA" == "true" && "$OS" == "Linux" ]]; then
    # Setup target library directory
    if [ "$BUILD_MODE" = "release" ]; then
        LIB_DIR="${PROJECT_DIR}/target/release"
    else
        LIB_DIR="${PROJECT_DIR}/target/debug"
    fi

    # If we built sherpa-onnx from source, copy libraries to target
    if [[ -n "${SHERPA_LIB_PATH:-}" && -d "${SHERPA_LIB_PATH}/lib" ]]; then
        log_info "Copying sherpa-onnx libraries to target directory..."
        mkdir -p "${LIB_DIR}"
        cp -av "${SHERPA_LIB_PATH}/lib/"*.so* "${LIB_DIR}/" 2>/dev/null || true
        log_info "Libraries copied to: ${LIB_DIR}"
    fi

    # Verify libraries exist in target directory
    if ls "${LIB_DIR}"/libsherpa-onnx-c-api.so* &>/dev/null 2>&1; then
        log_info "Shared libraries found in: ${LIB_DIR}"
    else
        log_warn "Shared libraries not found in ${LIB_DIR}"
        log_warn "The binary may fail to load at runtime"
    fi

    RUN_SCRIPT="${PROJECT_DIR}/run-voice-assistant.sh"
    cat > "$RUN_SCRIPT" << 'EOF'
#!/bin/bash
# Wrapper script to run voice-assistant with proper CUDA library paths
# This script sets up LD_LIBRARY_PATH so the binary can find its shared libraries

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Setup CUDA environment
if [[ -z "$CUDA_HOME" ]]; then
    if [[ -d /usr/local/cuda ]]; then
        export CUDA_HOME=/usr/local/cuda
    else
        for cuda_dir in /usr/local/cuda-*; do
            if [[ -d "$cuda_dir" ]]; then
                export CUDA_HOME="$cuda_dir"
                break
            fi
        done
    fi
fi

if [[ -n "$CUDA_HOME" ]]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}"
fi

# Also check for Jetson-specific paths
if [[ -d /usr/lib/aarch64-linux-gnu/tegra ]]; then
    export LD_LIBRARY_PATH="/usr/lib/aarch64-linux-gnu/tegra:$LD_LIBRARY_PATH"
fi

# Add the sherpa-onnx lib directory if it exists (in ~/.voice-assistant/rust/lib for portability)
if [[ -d "$HOME/.voice-assistant/rust/lib" ]]; then
    export LD_LIBRARY_PATH="$HOME/.voice-assistant/rust/lib:$LD_LIBRARY_PATH"
fi

# Add the target directory to library path for sherpa-onnx shared libs
EOF
    if [ "$BUILD_MODE" = "release" ]; then
        echo 'export LD_LIBRARY_PATH="$SCRIPT_DIR/target/release:${LD_LIBRARY_PATH:-}"' >> "$RUN_SCRIPT"
        echo 'exec "$SCRIPT_DIR/target/release/voice-assistant" "$@"' >> "$RUN_SCRIPT"
    else
        echo 'export LD_LIBRARY_PATH="$SCRIPT_DIR/target/debug:${LD_LIBRARY_PATH:-}"' >> "$RUN_SCRIPT"
        echo 'exec "$SCRIPT_DIR/target/debug/voice-assistant" "$@"' >> "$RUN_SCRIPT"
    fi
    chmod +x "$RUN_SCRIPT"

    log_info "Created run wrapper script for CUDA: ${RUN_SCRIPT}"
    echo ""
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo -e "${GREEN}CUDA Build Complete${NC}"
    echo -e "${GREEN}═══════════════════════════════════════════════════════════${NC}"
    echo ""
    if [[ -n "${SHERPA_LIB_PATH:-}" ]]; then
        echo "Built with sherpa-onnx compiled from source for CUDA ${CUDA_VERSION}."
        echo "Runtime libraries installed to: ~/.voice-assistant/rust/lib"
        echo "Libraries also copied to: ${LIB_DIR}"
    fi
    echo ""
    echo "The binary uses dynamically linked CUDA libraries."
    echo "Use the wrapper script to run with proper library paths:"
    echo ""
    echo "  ./run-voice-assistant.sh [options]"
    echo ""
    echo "For portable deployment, copy ~/.voice-assistant/rust and the binary to the target machine."
else
    echo "To run the assistant:"
    if [ "$BUILD_MODE" = "release" ]; then
        echo "  ./target/release/voice-assistant"
    else
        echo "  ./target/debug/voice-assistant"
    fi
fi
echo ""
echo "Or use cargo:"
if [ "$BUILD_MODE" = "release" ]; then
    echo "  cargo run --release"
else
    echo "  cargo run"
fi
