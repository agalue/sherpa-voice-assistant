#!/bin/bash

# --- Configuration ---
MAC_FILE="$HOME/.airpods_mac"
BT_CONF="/lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# --- Function: Print colored messages ---
log_info() { echo -e "${GREEN}[+]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[!]${NC} $1"; }
log_error() { echo -e "${RED}[✗]${NC} $1"; }

# --- Function: Initialize Bluetooth Controller ---
init_bluetooth() {
    log_info "Initializing Bluetooth controller..."
    sudo rfkill unblock bluetooth
    sudo systemctl start bluetooth
    sleep 2
    
    # Ensure bluetoothctl sees the controller
    if ! bluetoothctl list | grep -q "Controller"; then
        log_error "No Bluetooth controller found"
        return 1
    fi
    
    # Power on and set up agent
    bluetoothctl power on
    sleep 1
    bluetoothctl agent on
    bluetoothctl default-agent
    log_info "Bluetooth controller ready"
}

# --- Function: Check if device is paired and trusted ---
check_device_status() {
    local mac=$1
    local info=$(bluetoothctl info "$mac" 2>/dev/null)
    
    if [ -z "$info" ]; then
        echo "unknown"
        return
    fi
    
    local paired=$(echo "$info" | grep "Paired: yes")
    local trusted=$(echo "$info" | grep "Trusted: yes")
    local connected=$(echo "$info" | grep "Connected: yes")
    
    if [ -n "$connected" ]; then
        echo "connected"
    elif [ -n "$paired" ] && [ -n "$trusted" ]; then
        echo "ready"
    elif [ -n "$paired" ]; then
        echo "paired"
    else
        echo "unpaired"
    fi
}

# --- Function: Connect to already-paired device with retry ---
connect_device() {
    local mac=$1
    local max_attempts=5
    local attempt=1
    
    log_info "Attempting to connect to $mac..."
    
    # Brief scan to wake up the device (especially important for AirPods)
    log_info "Scanning to wake up device..."
    bluetoothctl --timeout 3 scan on &
    local scan_pid=$!
    sleep 3
    kill $scan_pid 2>/dev/null || true
    
    while [ $attempt -le $max_attempts ]; do
        log_info "Connection attempt $attempt/$max_attempts..."
        
        if bluetoothctl connect "$mac" 2>&1 | grep -q "Connection successful"; then
            log_info "Successfully connected!"
            return 0
        fi
        
        if [ $attempt -lt $max_attempts ]; then
            local wait_time=$((attempt * 2))
            log_warn "Connection failed, retrying in ${wait_time}s..."
            sleep $wait_time
        fi
        
        attempt=$((attempt + 1))
    done
    
    log_error "Failed to connect after $max_attempts attempts"
    return 1
}

# --- Function: Pair new device ---
pair_device() {
    local mac=$1
    
    log_info "Pairing with $mac..."
    bluetoothctl pair "$mac"
    
    log_info "Trusting device..."
    bluetoothctl trust "$mac"
    
    log_info "Connecting..."
    bluetoothctl connect "$mac"
}

# --- Function: Audio Profile Switcher ---
set_profile() {
    local profile=$1
    # Automatically find the AirPod card name even if the Index changes
    CARD=$(pactl list cards short | grep "bluez_card" | awk '{print $2}')
    
    if [ -z "$CARD" ]; then
        log_warn "No Bluetooth audio card found"
        return
    fi

    if [ "$profile" == "voice" ]; then
        log_info "Switching $CARD to AI Mode (HFP)..."
        pactl set-card-profile "$CARD" handsfree_head_unit
    else
        log_info "Switching $CARD to Music Mode (A2DP)..."
        pactl set-card-profile "$CARD" a2dp_sink
    fi
}

# --- Function: Battery Check ---
get_battery() {
    source "$MAC_FILE" 2>/dev/null
    if [[ -z "$AIRPODS_MAC" ]]; then 
        log_error "No AirPods paired"
        return
    fi
    
    # Format MAC for UPower (e.g., 00:11:22 -> 00_11_22)
    UP_MAC=${AIRPODS_MAC//:/_}
    DEVICE_PATH=$(upower -e | grep "bluez_device_$UP_MAC")
    
    if [ -n "$DEVICE_PATH" ]; then
        echo "--- AirPods Status ---"
        upower -i "$DEVICE_PATH" | grep -E "state|percentage|updated" | sed 's/^[ \t]*//'
    else
        log_warn "Battery data not available. Are the AirPods in your ears?"
    fi
}

case "$1" in
    setup)
        log_info "Applying NVIDIA Bluetooth Audio Patch..."
        sudo sed -i 's/--noplugin=audio,a2dp,avrcp//g' "$BT_CONF"
        sudo systemctl daemon-reload
        sudo apt-get update && sudo apt-get install -y pulseaudio-module-bluetooth bluez-tools upower
        log_info "Setup complete. Please reboot your Jetson."
        ;;

    wizard)
        init_bluetooth || exit 1
        
        log_info "Put AirPods in pairing mode (press and hold button until LED flashes white)..."
        sleep 2
        
        bluetoothctl --timeout 15 scan on &
        scan_pid=$!
        sleep 10
        kill $scan_pid 2>/dev/null || true
        
        mapfile -t devices < <(bluetoothctl devices | grep -i "AirPod")
        if [ ${#devices[@]} -eq 0 ]; then 
            log_error "No AirPods found"
            exit 1
        fi
        
        for i in "${!devices[@]}"; do 
            echo "$((i+1))) ${devices[$i]}"
        done
        read -p "Select number: " choice
        ADDR=$(echo "${devices[$((choice-1))]}" | awk '{print $2}')
        echo "export AIRPODS_MAC=\"$ADDR\"" > "$MAC_FILE"
        
        pair_device "$ADDR"
        ;;

    connect)
        if [ ! -f "$MAC_FILE" ]; then
            log_error "No device MAC address found. Run 'wizard' first."
            exit 1
        fi
        
        source "$MAC_FILE"
        
        # Initialize Bluetooth
        init_bluetooth || exit 1
        
        # Check device status
        status=$(check_device_status "$AIRPODS_MAC")
        log_info "Device status: $status"
        
        case "$status" in
            "connected")
                log_info "Already connected!"
                ;;
            "ready")
                # Device is paired and trusted, just connect
                connect_device "$AIRPODS_MAC" || exit 1
                ;;
            "paired")
                # Device is paired but not trusted
                log_info "Device paired but not trusted. Trusting now..."
                bluetoothctl trust "$AIRPODS_MAC"
                connect_device "$AIRPODS_MAC" || exit 1
                ;;
            "unpaired"|"unknown")
                log_error "Device not paired. Run 'wizard' first."
                exit 1
                ;;
        esac
        
        sleep 3 # Wait for PulseAudio to register
        set_profile "voice"
        log_info "Connected in AI Voice Mode"
        get_battery
        ;;

    battery)
        get_battery
        ;;

    mode-voice|mode-music)
        set_profile "${1#mode-}"
        ;;

    disconnect)
        source "$MAC_FILE"
        bluetoothctl disconnect "$AIRPODS_MAC"
        sudo systemctl stop bluetooth
        sudo rfkill block bluetooth
        log_info "Jetson Bluetooth Offline (Power Saved)"
        ;;

    *)
        echo "Usage: $0 {setup|wizard|connect|battery|mode-voice|mode-music|disconnect}"
        echo ""
        echo "Commands:"
        echo "  setup      - Install dependencies and configure Bluetooth"
        echo "  wizard     - Pair new AirPods (only needed once)"
        echo "  connect    - Connect to already-paired AirPods (idempotent)"
        echo "  battery    - Check AirPods battery level"
        echo "  mode-voice - Switch to voice mode (HFP)"
        echo "  mode-music - Switch to music mode (A2DP)"
        echo "  disconnect - Disconnect and disable Bluetooth"
        ;;
esac
