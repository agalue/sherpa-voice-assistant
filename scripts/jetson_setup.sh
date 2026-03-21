#!/bin/bash
# Quick start script for voice assistant on Jetson Orin Nano
# This script helps manage Ollama and SearXNG services

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SEARXNG_DIR="${REPO_ROOT}/searxng"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Helper functions
print_status() {
    echo -e "${BLUE}==>${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check if running on Jetson
check_jetson() {
    if [ ! -f /etc/nv_tegra_release ]; then
        print_warning "This script is designed for NVIDIA Jetson devices"
        read -p "Continue anyway? (y/N) " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    fi
}

# Configure Ollama service
configure_ollama() {
    print_status "Configuring Ollama systemd service..."
    
    # Check if ollama service exists
    if ! systemctl list-unit-files | grep -q ollama.service; then
        print_error "Ollama service not found. Please install Ollama first."
        echo "Run: curl -fsSL https://ollama.com/install.sh | sh"
        exit 1
    fi
    
    # Create systemd override
    print_status "Creating systemd override for Ollama..."
    sudo mkdir -p /etc/systemd/system/ollama.service.d
    
    cat << EOF | sudo tee /etc/systemd/system/ollama.service.d/override.conf > /dev/null
[Service]
Environment="OLLAMA_NUM_PARALLEL=1"
Environment="OLLAMA_MAX_LOADED_MODELS=1"
Environment="OLLAMA_NUM_CTX=1024"
Environment="OLLAMA_KEEP_ALIVE=-1"
Environment="OLLAMA_MAX_VRAM=3221225472"
Environment="OLLAMA_HOST=${JETSON_OLLAMA_HOST:-127.0.0.1:11434}"
EOF
    
    # Reload and restart
    print_status "Reloading systemd and restarting Ollama..."
    sudo systemctl daemon-reload
    sudo systemctl restart ollama
    
    # Check status
    if systemctl is-active --quiet ollama; then
        print_success "Ollama configured and running"
    else
        print_error "Ollama failed to start. Check logs with: sudo journalctl -u ollama -n 50"
        exit 1
    fi
}

# Pull optimized model
pull_model() {
    local model="${1:-qwen2.5:1.5b}"
    
    print_status "Pulling model: ${model}..."
    
    if ollama pull "$model"; then
        print_success "Model ${model} downloaded successfully"
    else
        print_error "Failed to pull model. Check your internet connection."
        exit 1
    fi
}

# Pre-load Ollama model to allocate GPU memory before voice assistant starts
preload_model() {
    local model="${1:-qwen2.5:1.5b}"
    
    print_status "Pre-loading model ${model} with num_ctx=1024 to reserve GPU memory..."
    
    if curl -s http://localhost:11434/api/generate -d "{
        \"model\": \"${model}\",
        \"prompt\": \".\",
        \"stream\": false,
        \"options\": {\"num_ctx\": 1024, \"num_predict\": 1}
    }" > /dev/null 2>&1; then
        print_success "Model ${model} pre-loaded successfully"
    else
        print_error "Failed to pre-load model. Is Ollama running?"
        exit 1
    fi
}

# Start SearXNG
start_searxng() {
    print_status "Starting SearXNG with Docker Compose..."
    
    if [ ! -f "${SEARXNG_DIR}/docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in ${SEARXNG_DIR}"
        exit 1
    fi
    
    cd "${SEARXNG_DIR}"
    
    if sudo docker compose up -d; then
        print_success "SearXNG started"
        print_status "Waiting for SearXNG to be ready..."
        sleep 3
        
        # Test if SearXNG is responding
        if curl -s http://localhost:8080/search?q=test&format=json > /dev/null; then
            print_success "SearXNG is responding"
        else
            print_warning "SearXNG may not be ready yet. Check logs with: cd searxng && docker compose logs"
        fi
    else
        print_error "Failed to start SearXNG"
        exit 1
    fi
    
    cd "${REPO_ROOT}"
}

# Stop SearXNG
stop_searxng() {
    print_status "Stopping SearXNG..."
    
    if [ ! -f "${SEARXNG_DIR}/docker-compose.yml" ]; then
        print_error "docker-compose.yml not found in ${SEARXNG_DIR}"
        exit 1
    fi
    
    cd "${SEARXNG_DIR}"
    
    if sudo docker compose down; then
        print_success "SearXNG stopped"
    else
        print_error "Failed to stop SearXNG"
        exit 1
    fi
    
    cd "${REPO_ROOT}"
}

# Show status
show_status() {
    print_status "System Status:"
    echo
    
    # Ollama status
    echo -n "Ollama Service: "
    if systemctl is-active --quiet ollama; then
        print_success "Running"
    else
        print_error "Not running"
    fi
    
    # Check if models are loaded
    if curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
        echo -n "Ollama API: "
        print_success "Responding"
        
        # List loaded models
        echo
        print_status "Available models:"
        curl -s http://localhost:11434/api/tags | jq -r '.models[] | "  - \(.name) (\(.size / 1024 / 1024 / 1024 | floor)GB)"' 2>/dev/null || echo "  (jq not installed, run: sudo apt install jq)"
    else
        echo -n "Ollama API: "
        print_error "Not responding"
    fi
    
    echo
    
    # SearXNG status
    echo -n "SearXNG: "
    if sudo docker ps | grep -q searxng; then
        print_success "Running"
        
        # Check if responding
        if curl -s http://localhost:8080/search?q=test&format=json > /dev/null 2>&1; then
            echo -n "SearXNG API: "
            print_success "Responding"
        else
            echo -n "SearXNG API: "
            print_warning "Not responding"
        fi
    else
        print_error "Not running"
    fi
    
    echo
    
    # Memory status
    print_status "Memory Usage:"
    free -h | grep -E "Mem:|Swap:"
    
    echo
    
    # GPU memory
    if command -v tegrastats &> /dev/null; then
        print_status "GPU Status:"
        timeout 1 tegrastats | head -1
    fi
}

# Check logs
check_logs() {
    local service="${1:-both}"
    
    case "$service" in
        ollama)
            print_status "Ollama logs (last 50 lines):"
            sudo journalctl -u ollama -n 50 --no-pager
            ;;
        searxng)
            print_status "SearXNG logs:"
            cd "${SEARXNG_DIR}"
            sudo docker compose logs --tail=50
            cd "${REPO_ROOT}"
            ;;
        both|*)
            print_status "Ollama logs (last 30 lines):"
            sudo journalctl -u ollama -n 30 --no-pager
            echo
            print_status "SearXNG logs (last 30 lines):"
            cd "${SEARXNG_DIR}"
            sudo docker compose logs --tail=30
            cd "${REPO_ROOT}"
            ;;
    esac
}

# Main script
show_help() {
    cat << EOF
Voice Assistant Jetson Management Script

Usage: ./scripts/jetson_setup.sh [command]

Commands:
    setup           Configure Ollama and start services (run once)
    configure       Configure Ollama systemd service only
    pull [model]    Pull model (default: qwen2.5:1.5b)
    preload [model] Pre-load model to reserve GPU memory
    start           Start SearXNG
    stop            Stop SearXNG
    restart         Restart both Ollama and SearXNG
    status          Show status of all services
    logs [service]  Show logs (ollama|searxng|both)
    help            Show this help message

Examples:
    # Initial setup
    ./scripts/jetson_setup.sh setup
    
    # Manual model management
    ./scripts/jetson_setup.sh pull qwen2.5:3b-instruct-q2_k
    ./scripts/jetson_setup.sh preload qwen2.5:3b-instruct-q2_k
    
    # Running voice assistant (auto-detects Jetson, pre-loads model)
    ./run-voice-assistant.sh -ollama-model qwen2.5:1.5b -stt-model tiny
    
    # Or use 3b model
    ./run-voice-assistant.sh -ollama-model qwen2.5:3b-instruct-q2_k -stt-model tiny
    
    # Check status
    ./scripts/jetson_setup.sh status
    ./scripts/jetson_setup.sh logs ollama

EOF
}

# Parse command
case "${1:-help}" in
    setup)
        check_jetson
        configure_ollama
        pull_model "${2:-qwen2.5:1.5b}"
        start_searxng
        echo
        show_status
        echo
        print_success "Setup complete!"
        echo
        echo "To run voice assistant (auto-detects Jetson, pre-loads model):"
        echo "  ./run-voice-assistant.sh -ollama-model qwen2.5:1.5b -stt-model tiny"
        echo
        echo "Or with 3b model:"
        echo "  ./run-voice-assistant.sh -ollama-model qwen2.5:3b-instruct-q2_k -stt-model tiny"
        ;;
    configure)
        check_jetson
        configure_ollama
        ;;
    pull)
        pull_model "${2:-qwen2.5:1.5b}"
        ;;
    preload)
        preload_model "${2:-qwen2.5:1.5b}"
        ;;
    start)
        start_searxng
        ;;
    stop)
        stop_searxng
        ;;
    restart)
        print_status "Restarting services..."
        sudo systemctl restart ollama
        stop_searxng
        sleep 2
        start_searxng
        print_success "Services restarted"
        ;;
    status)
        show_status
        ;;
    logs)
        check_logs "${2:-both}"
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        echo
        show_help
        exit 1
        ;;
esac
