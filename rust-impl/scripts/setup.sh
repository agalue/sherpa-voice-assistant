#!/bin/bash
#
# Rust Voice Assistant Setup - Wrapper Script
# This script delegates to the shared Go setup.sh script for model downloads.
# Both Go and Rust implementations share the same models (~/.voice-assistant/models)
#

set -e

# Determine script locations
RUST_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUST_PROJECT_DIR="$(dirname "$RUST_SCRIPT_DIR")"
GO_PROJECT_DIR="$(dirname "$RUST_PROJECT_DIR")"
GO_SETUP_SCRIPT="${GO_PROJECT_DIR}/scripts/setup.sh"

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  Rust Voice Assistant - Setup Wrapper${NC}"
echo -e "${BLUE}═══════════════════════════════════════════════════════════${NC}"
echo

# Verify Go setup script exists
if [[ ! -f "$GO_SETUP_SCRIPT" ]]; then
    echo -e "${RED}Error: Shared setup script not found at:${NC}"
    echo -e "${RED}  ${GO_SETUP_SCRIPT}${NC}"
    echo
    echo -e "${RED}Expected project structure:${NC}"
    echo "  sherpa-voice-assistant/"
    echo "  ├── scripts/setup.sh        (shared setup script)"
    echo "  └── rust-impl/"
    echo "      └── scripts/setup.sh    (this wrapper)"
    exit 1
fi

echo -e "${GREEN}✓ Found shared setup script${NC}"
echo -e "${BLUE}  Delegating to: ${GO_SETUP_SCRIPT}${NC}"
echo

# Pass all arguments to the shared Go setup script
exec "$GO_SETUP_SCRIPT" "$@"
