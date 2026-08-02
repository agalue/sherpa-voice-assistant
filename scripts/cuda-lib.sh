#!/bin/bash
# cuda-lib.sh — shared CUDA utility functions for the voice assistant build scripts.
#
# Sourced by both scripts/build.sh (Go) and rust-impl/scripts/build.sh (Rust).
# Do not execute directly.
#
# Requires the following variables to be defined by the caller before sourcing:
#   GREEN, YELLOW, RED, NC   — ANSI color codes
#
# All functions print to stdout unless noted. Functions used as command
# substitution (install_cuda12_for_aarch64) route all log output to stderr
# and emit only a path to stdout.

# Detect an NVIDIA GPU — discrete or Jetson SoC.
detect_nvidia_gpu() {
    for path in /usr/bin/nvidia-smi /usr/local/bin/nvidia-smi /opt/nvidia/bin/nvidia-smi; do
        [[ -f "$path" ]] && return 0
    done

    [[ -e /dev/nvidia0 ]] && return 0

    for path in /dev/nvhost-gpu /dev/nvhost-ctrl-gpu /dev/nvmap /etc/nv_tegra_release \
        /sys/devices/gpu.0 /sys/devices/17000000.ga10b /sys/devices/17000000.gv11b; do
        [[ -e "$path" ]] && return 0
    done

    if [[ -f /proc/device-tree/compatible ]]; then
        grep -q "nvidia,tegra\|nvidia,jetson" /proc/device-tree/compatible 2>/dev/null && return 0
    fi

    return 1
}

# Return 0 if any CUDA toolkit installation is found.
check_cuda_toolkit() {
    command -v nvcc &>/dev/null && return 0
    [[ -d /usr/local/cuda ]] && return 0
    for cuda_dir in /usr/local/cuda-*; do
        [[ -d "$cuda_dir" ]] && return 0
    done
    return 1
}

# Print the detected CUDA version as "major.minor" (e.g. "12.6"), or empty string.
get_cuda_version() {
    local cuda_version=""

    if command -v nvcc &>/dev/null; then
        cuda_version=$(nvcc --version 2>/dev/null | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
    elif [[ -f /usr/local/cuda/version.txt ]]; then
        cuda_version=$(grep -oP 'CUDA Version \K[0-9]+\.[0-9]+' /usr/local/cuda/version.txt)
    elif [[ -f /usr/local/cuda/version.json ]]; then
        cuda_version=$(grep -oP '"cuda" *: *\{ *"version" *: *"\K[0-9]+\.[0-9]+' \
            /usr/local/cuda/version.json 2>/dev/null || true)
    fi

    # Last resort: resolve the /usr/local/cuda symlink to extract the version.
    if [[ -z "$cuda_version" && -d /usr/local/cuda ]]; then
        local target
        target=$(readlink -f /usr/local/cuda 2>/dev/null || true)
        if [[ "$target" =~ cuda-([0-9]+\.[0-9]+) ]]; then
            cuda_version="${BASH_REMATCH[1]}"
        fi
    fi

    echo "$cuda_version"
}

# Print the ONNX Runtime version required for a given CUDA version on aarch64 Jetson.
# See: https://github.com/k2-fsa/sherpa-onnx/blob/main/cmake/onnxruntime-linux-aarch64-gpu.cmake
get_onnxruntime_version_for_cuda() {
    local cuda_ver="$1"
    local cuda_major="${cuda_ver%%.*}"

    case "$cuda_ver" in
    10.2*)  echo "1.11.0" ;;  # Jetson Nano B01 / JetPack 4.x
    11.4*)  echo "1.16.0" ;;  # Jetson Orin NX / JetPack 5.x
    11.*)   echo "1.16.0" ;;  # Generic CUDA 11
    12.2*)  echo "1.18.0" ;;  # CUDA 12.2, cuDNN8
    12.*)   echo "1.18.1" ;;  # JetPack 6.2+ (12.6 cuDNN9) and any other CUDA 12
    13.*)   echo "1.18.1" ;;  # JetPack 7.2+ — redirected to CUDA 12.6 build by install_cuda12_for_aarch64
    *)
        if [[ "$cuda_major" -ge 12 ]]; then
            echo "1.18.1"
        else
            echo "1.16.0"
        fi
        ;;
    esac
}

# Install the CUDA 12.6 toolkit on aarch64 Jetson systems running CUDA 13 (JetPack 7.2+).
#
# JetPack 7.2 ships CUDA 13, but no pre-built ONNX Runtime aarch64 GPU binary exists for
# CUDA 13 yet. Building sherpa-onnx against CUDA 12.6 works because the CUDA 13
# nvgpu driver on Jetson natively handles CUDA 12-compiled code without any compat shim.
# (The cuda-compat-12-6 shim is for discrete Tesla/datacenter GPUs only and must NOT
# be loaded on Jetson — it breaks cudaSetDevice with error 801.)
#
# cuda-toolkit-12-6 and cuda-compat-12-6 are only published in the NVIDIA Ubuntu 22.04
# arm64 repo, not in the Ubuntu 24.04 repo. On Ubuntu 24.04 Jetson systems we add the
# ubuntu2204/arm64 NVIDIA repo as a secondary apt source. The existing keyring installed
# by cuda-keyring (cuda-archive-keyring.gpg) covers both repos, so no new GPG key is
# needed. The two repos have no package name overlap (12.x vs 13.x), so adding the
# ubuntu2204 source does not risk unintended upgrades to the CUDA 13 installation.
#
# This function is idempotent: exits immediately if cuda-toolkit-12-6 is already present.
# All log output goes to stderr; stdout emits only the install path (/usr/local/cuda-12.6).
install_cuda12_for_aarch64() {
    if [[ -d /usr/local/cuda-12.6 ]] && command -v /usr/local/cuda-12.6/bin/nvcc &>/dev/null; then
        echo -e "${GREEN}CUDA 12.6 toolkit already present at /usr/local/cuda-12.6${NC}" >&2
        echo "/usr/local/cuda-12.6"
        return 0
    fi

    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2
    echo -e "${YELLOW}JetPack 7.2+ detected (CUDA 13). Installing CUDA 12.6 toolkit${NC}" >&2
    echo -e "${YELLOW}so sherpa-onnx can be built against ONNX Runtime 1.18.1.${NC}" >&2
    echo -e "${YELLOW}The CUDA 13 driver will execute CUDA 12-compiled code natively.${NC}" >&2
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}" >&2

    # On Ubuntu 24.04, cuda-toolkit-12-6 is absent from the ubuntu2404 NVIDIA repo.
    # It exists in the ubuntu2204/arm64 repo. We add that repo as a secondary source
    # using the keyring already installed by cuda-keyring (cuda-archive-keyring.gpg).
    local sources_file="/etc/apt/sources.list.d/cuda-ubuntu2204-arm64.list"
    local keyring_path="/usr/share/keyrings/cuda-archive-keyring.gpg"

    if ! apt-cache show cuda-toolkit-12-6 &>/dev/null 2>&1; then
        echo -e "${YELLOW}cuda-toolkit-12-6 not found in current repos.${NC}" >&2

        # Ensure the NVIDIA keyring is present. The cuda-keyring package (already
        # installed for the ubuntu2404 repo) provides the same GPG key for all NVIDIA
        # CUDA repos, so we only need to install it if it's completely absent.
        if [[ ! -f "$keyring_path" ]]; then
            echo -e "${YELLOW}Installing NVIDIA CUDA apt keyring...${NC}" >&2
            local keyring_deb="/tmp/cuda-keyring_1.1-1_all.deb"
            wget -q -O "$keyring_deb" \
                "https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/arm64/cuda-keyring_1.1-1_all.deb" >&2
            sudo dpkg -i "$keyring_deb" >&2
            rm -f "$keyring_deb"
        fi

        # Add the ubuntu2204/arm64 repo if not already present.
        if [[ ! -f "$sources_file" ]]; then
            echo -e "${YELLOW}Adding NVIDIA CUDA ubuntu2204/arm64 repo (for CUDA 12.6 packages)...${NC}" >&2
            echo "deb [signed-by=${keyring_path}] https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/arm64/ /" \
                | sudo tee "$sources_file" >/dev/null
        fi

        # No apt pinning needed: the ubuntu2204 and ubuntu2404 NVIDIA repos contain
        # different CUDA major versions (12.x vs 13.x) with no package name overlap,
        # so they cannot accidentally upgrade each other.

        echo -e "${YELLOW}Updating apt package lists...${NC}" >&2
        sudo apt-get update -q >&2
    fi

    echo -e "${YELLOW}Installing cuda-toolkit-12-6 and cuda-compat-12-6...${NC}" >&2
    sudo apt-get install -y cuda-toolkit-12-6 cuda-compat-12-6 >&2

    if [[ ! -d /usr/local/cuda-12.6 ]]; then
        echo -e "${RED}ERROR: cuda-toolkit-12-6 installation failed.${NC}" >&2
        echo -e "${RED}The ubuntu2204/arm64 NVIDIA repo was added to apt but the package${NC}" >&2
        echo -e "${RED}could not be installed. Check apt output above for details.${NC}" >&2
        return 1
    fi

    echo -e "${GREEN}CUDA 12.6 toolkit installed at /usr/local/cuda-12.6${NC}" >&2
    echo "/usr/local/cuda-12.6"
}
