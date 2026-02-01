#!/bin/bash

# --- Configuration ---
MAC_FILE="$HOME/.airpods_mac"
BT_CONF="/lib/systemd/system/bluetooth.service.d/nv-bluetooth-service.conf"

# --- Function: Audio Profile Switcher ---
set_profile() {
    local profile=$1
    # Automatically find the AirPod card name even if the Index changes
    CARD=$(pactl list cards short | grep "bluez_card" | awk '{print $2}')
    
    if [ -z "$CARD" ]; then
        echo "[!] No Bluetooth audio card found."
        return
    fi

    if [ "$profile" == "voice" ]; then
        echo "[*] Switching $CARD to AI Mode (HFP)..."
        pactl set-card-profile "$CARD" handsfree_head_unit
    else
        echo "[*] Switching $CARD to Music Mode (A2DP)..."
        pactl set-card-profile "$CARD" a2dp_sink
    fi
}

# --- Function: Battery Check ---
get_battery() {
    source "$MAC_FILE" 2>/dev/null
    if [[ -z "$AIRPODS_MAC" ]]; then echo "Error: No AirPods paired."; return; fi
    
    # Format MAC for UPower (e.g., 00:11:22 -> 00_11_22)
    UP_MAC=${AIRPODS_MAC//:/_}
    DEVICE_PATH=$(upower -e | grep "bluez_device_$UP_MAC")
    
    if [ -n "$DEVICE_PATH" ]; then
        echo "--- AirPods Pro 2 Status ---"
        upower -i "$DEVICE_PATH" | grep -E "state|percentage|updated" | sed 's/^[ \t]*//'
    else
        echo "[!] Battery data not available. Are the AirPods in your ears?"
    fi
}

case "$1" in
    setup)
        echo "[*] Applying NVIDIA Bluetooth Audio Patch..."
        sudo sed -i 's/--noplugin=audio,a2dp,avrcp//g' "$BT_CONF"
        sudo systemctl daemon-reload
        sudo apt-get update && sudo apt-get install -y pulseaudio-module-bluetooth bluez-tools upower
        echo "[+] Setup complete. Please reboot your Jetson."
        ;;

    wizard)
        echo "[*] Put AirPods in pairing mode..."
        bluetoothctl --timeout 15 scan on
        mapfile -t devices < <(bluetoothctl devices | grep -i "AirPod")
        if [ ${#devices[@]} -eq 0 ]; then echo "No AirPods found."; exit 1; fi
        for i in "${!devices[@]}"; do echo "$((i+1))) ${devices[$i]}"; done
        read -p "Select number: " choice
        ADDR=$(echo "${devices[$((choice-1))]}" | awk '{print $2}')
        echo "export AIRPODS_MAC=\"$ADDR\"" > "$MAC_FILE"
        bluetoothctl pair "$ADDR" && bluetoothctl trust "$ADDR" && bluetoothctl connect "$ADDR"
        ;;

    connect)
        source "$MAC_FILE"
        sudo rfkill unblock bluetooth
        sudo systemctl start bluetooth
        sleep 1
        bluetoothctl connect "$AIRPODS_MAC"
        sleep 3 # Wait for PulseAudio to register
        set_profile "voice"
        echo "[+] Connected in AI Voice Mode."
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
        echo "[+] Jetson Bluetooth Offline (Power Saved)."
        ;;

    *)
        echo "Usage: ./manage_pods.sh {setup|wizard|connect|battery|mode-voice|mode-music|disconnect}"
        ;;
esac
