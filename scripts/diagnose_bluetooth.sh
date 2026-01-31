#!/bin/bash
# Bluetooth Audio Diagnostics for Jetson Nano
# Run this script to gather information about your Bluetooth audio setup

echo "========================================"
echo "Bluetooth Audio Diagnostic Tool"
echo "========================================"
echo ""

# Check if running on Linux
if [[ "$OSTYPE" != "linux-gnu"* ]]; then
    echo "⚠️  This script is designed for Linux (Jetson Nano)"
    echo "   Current OS: $OSTYPE"
    echo ""
fi

echo "1. Audio Devices"
echo "----------------"
if command -v aplay &> /dev/null; then
    echo "Available playback devices:"
    aplay -l
    echo ""
else
    echo "⚠️  aplay not found (install alsa-utils)"
    echo ""
fi

echo "2. Bluetooth Devices"
echo "--------------------"
if command -v bluetoothctl &> /dev/null; then
    echo "Connected Bluetooth devices:"
    bluetoothctl devices | head -n 10
    echo ""
else
    echo "⚠️  bluetoothctl not found"
    echo ""
fi

echo "3. PulseAudio Sinks (if available)"
echo "----------------------------------"
if command -v pactl &> /dev/null; then
    echo "Active audio sinks:"
    pactl list short sinks
    echo ""
    echo "Bluetooth sink details:"
    pactl list sinks | grep -A 20 -i "bluetooth\|airpods" | head -n 30
    echo ""
else
    echo "⚠️  PulseAudio not found (pactl)"
    echo ""
fi

echo "4. Current Audio Hardware Parameters"
echo "-------------------------------------"
if [ -d "/proc/asound" ]; then
    echo "Active PCM streams:"
    for hw_params in /proc/asound/card*/pcm*/sub*/hw_params; do
        if [ -f "$hw_params" ] && [ -s "$hw_params" ]; then
            echo "Device: $(dirname $hw_params)"
            cat "$hw_params"
            echo ""
        fi
    done
    
    if ! ls /proc/asound/card*/pcm*/sub*/hw_params 2>/dev/null | grep -q .; then
        echo "No active PCM streams (start playback to see parameters)"
    fi
    echo ""
else
    echo "⚠️  /proc/asound not found"
    echo ""
fi

echo "5. Bluetooth Audio Profile"
echo "--------------------------"
if command -v pactl &> /dev/null; then
    echo "Current Bluetooth profile (A2DP recommended for playback):"
    pactl list cards | grep -A 30 "bluez" | grep -E "Active Profile|Profiles:" | head -n 10
    echo ""
else
    echo "⚠️  Cannot check Bluetooth profile (pactl not available)"
    echo ""
fi

echo "6. System Audio Configuration"
echo "------------------------------"
if [ -f "/etc/asound.conf" ]; then
    echo "Global ALSA config (/etc/asound.conf):"
    cat /etc/asound.conf
    echo ""
else
    echo "No /etc/asound.conf found"
fi

if [ -f "$HOME/.asoundrc" ]; then
    echo "User ALSA config (~/.asoundrc):"
    cat "$HOME/.asoundrc"
    echo ""
else
    echo "No ~/.asoundrc found"
fi
echo ""

echo "7. Recommended Actions"
echo "----------------------"
echo "✅ Ensure AirPods Pro are in A2DP mode (high quality audio)"
echo "✅ Check sample rate: Bluetooth devices typically use 48000 Hz"
echo "✅ Test with speaker-test: speaker-test -t wav -c 1 -r 48000"
echo "✅ If using PulseAudio, restart it: pulseaudio --kill && pulseaudio --start"
echo ""

echo "8. Quick Audio Test"
echo "-------------------"
echo "Testing audio playback to default device..."

if command -v speaker-test &> /dev/null; then
    echo "Running 2-second test tone at 48000 Hz..."
    timeout 2 speaker-test -t sine -f 440 -c 1 -r 48000 2>&1 | head -n 10
    echo ""
    echo "Did you hear a clear tone? If distorted, there may be a Bluetooth issue."
else
    echo "⚠️  speaker-test not found (install alsa-utils)"
fi
echo ""

echo "========================================"
echo "Diagnostic Complete"
echo "========================================"
echo ""
echo "💡 Tips for fixing distorted audio:"
echo "   1. Verify Bluetooth is using A2DP profile (not HSP/HFP)"
echo "   2. Ensure device sample rate is 48000 Hz"
echo "   3. Increase audio buffer size (reduce underruns)"
echo "   4. Check dmesg for audio errors: dmesg | grep -i audio"
echo ""
