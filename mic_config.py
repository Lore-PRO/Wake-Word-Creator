
# -------------------------------------------------------------------------------
# Project: Wake Word Creator
# File: mic_config.py
# Author: Lorenzo Prometti (@Lore-PRO)
# License: CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 
# 4.0 International License. To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc/4.0/
#
# For commercial licensing inquiries, please contact: lorenzo.prometti@gmail.com
# -------------------------------------------------------------------------------


import pyaudio

# Defines the priority hierarchy for input devices
PRIORITY_DEVICES = [
    "Mac",
    "AirPods",
    "iPhone",
    "BlackHole"
]

def get_best_device():
    """
    Scans available audio devices and selects the best one based on the
    priority list. Returns BOTH the index and the name.
    """
    p = pyaudio.PyAudio()
    found_devices = {}
    
    for i in range(p.get_device_count()):
        try:
            info = p.get_device_info_by_index(i)
            if info["maxInputChannels"] > 0:
                name = info["name"]
                found_devices[i] = name
                print(f"🎤 Detected: [{i}] {name}")
        except Exception as e:
            print(f"⚠️ Error accessing device {i}: {e}")

    p.terminate()

    # Search for priority match
    for pref_name in PRIORITY_DEVICES:
        for index, actual_name in found_devices.items():
            if pref_name.lower() in actual_name.lower():
                print(f"✅ Selected device: {actual_name} (Index: {index})")
                return index, actual_name
    
    # Fallback: Try to use device 0 if available
    if 0 in found_devices:
        print(f"⚠️ No preferred device found. Using default: {found_devices[0]} (Index: 0)")
        return 0, found_devices[0]
        
    print("⚠️ No input devices found!")
    return 0, "Unknown Device"

# Global variables loaded once upon import
# The tuple is unpacked so other scripts can import only the necessary constants
DEVICE_INDEX, DEVICE_NAME = get_best_device()
