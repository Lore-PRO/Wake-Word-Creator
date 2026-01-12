
# -------------------------------------------------------------------------------
# Project: Wake Word Creator
# File: record_positives.py
# Author: Lorenzo Prometti (@Lore-PRO)
# License: CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 
# 4.0 International License. To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc/4.0/
#
# For commercial licensing inquiries, please contact: lorenzo.prometti@gmail.com
# -------------------------------------------------------------------------------


import os
import shutil
import sounddevice as sd
import soundfile as sf
import numpy as np
import time
import threading
import sys
import glob
from mic_config import DEVICE_INDEX, DEVICE_NAME


# --- CONFIGURATION ---
SAMPLE_RATE = 16000
CHANNELS = 1
DURATION = 1.0
BASE_DATASET_PATH = "dataset"
OUTPUT_DIR = os.path.join(BASE_DATASET_PATH, "positives")
FILE_PREFIX = "positive_"


# --- TARGETS ---
MIN_POSITIVES = 30     # Required for training
IDEAL_POSITIVES = 50   # Better generalization


# --- HELPER FUNCTIONS ---
def get_next_index(directory, prefix):
    """Scans the directory to find the highest existing index."""
    if not os.path.exists(directory): return 0
    files = [f for f in os.listdir(directory) if f.startswith(prefix) and f.endswith(".wav")]
    if not files: return 0
    highest_idx = -1
    for f in files:
        try:
            num_part = f.replace(prefix, "").replace(".wav", "")
            idx = int(num_part)
            if idx > highest_idx: highest_idx = idx
        except ValueError: continue
    return highest_idx + 1


# --- STEP 1: INITIAL STATUS CHECK ---
os.system('cls' if os.name == 'nt' else 'clear')
print(f"🎙  Selected Input Device: [{DEVICE_INDEX}] {DEVICE_NAME}")
print("\n--- 🧠 SMART DATASET ADVISOR (POSITIVES) ---")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

num_positives_existing = len(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))

print(f"📊 Current Status:")
print(f"   👉 Positives found: {num_positives_existing}")


# --- STEP 2: CLEANUP DECISION (Must happen BEFORE diagnosis) ---
start_index = 0
if num_positives_existing > 0:
    should_clear = input(f"\n🗑  Do you want to clear the existing '{OUTPUT_DIR}' folder and start fresh? (y/N): ").strip().lower()
    
    if should_clear == 'y':
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)
        print(f"🧹 Folder cleared. Resetting count to 0.")
        num_positives_existing = 0
    else:
        print(f"📂 Keeping existing files.")
        start_index = get_next_index(OUTPUT_DIR, FILE_PREFIX)
else:
    start_index = 0


# --- STEP 3: DIAGNOSIS & TARGETS ---
print(f"\n🎯 TARGETS:")
print(f"   ⚠️ Minimum: {MIN_POSITIVES} samples (Required for training)")
print(f"   ✅ Ideal:   {IDEAL_POSITIVES} samples (Better generalization)")

if num_positives_existing < MIN_POSITIVES:
    gap = MIN_POSITIVES - num_positives_existing
    print(f"\n❌ DIAGNOSIS: INSUFFICIENT DATA. You need at least {gap} more samples.")
    suggested_samples = gap
    prompt_text = f"\n🔢 Enter number of samples to ADD [Press ENTER for Default: {suggested_samples}]: "

elif num_positives_existing < IDEAL_POSITIVES:
    gap = IDEAL_POSITIVES - num_positives_existing
    print(f"\n⚠️  DIAGNOSIS: ACCEPTABLE ({num_positives_existing} samples).")
    print(f"   Recommendation: Add {gap} more to reach Rock-Solid quality, or press ENTER to skip.")
    suggested_samples = 0
    prompt_text = f"\n🔢 Enter number of samples to ADD [Press ENTER to Skip]: "

else:
    print(f"\n✅ DIAGNOSIS: ROCK-SOLID ({num_positives_existing} samples). You have enough data.")
    suggested_samples = 0
    prompt_text = f"\n🔢 Enter number of samples to ADD [Press ENTER to Skip]: "


# --- STEP 4: INPUT HANDLING ---
try:
    user_input = input(prompt_text)
    
    if user_input.strip() == "":
        NUM_SAMPLES = suggested_samples
    else:
        NUM_SAMPLES = int(user_input)

    if NUM_SAMPLES == 0:
        print("👋 Session finished without recording.")
        exit()

except ValueError:
    print("❌ Invalid input.")
    exit()


print(f"✅ Target: Recording {NUM_SAMPLES} new samples starting from index {start_index}.\n")

def record_sample(filename):
    print(f"\n--- 🎤 {os.path.basename(filename)} ---")
    input("Press ENTER to record...")

    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=CHANNELS, device=DEVICE_INDEX)
    
    def visual_bar():
        steps = 20
        for i in range(1, steps + 1):
            time.sleep(DURATION / steps)
            bar = "█" * i + " " * (steps - i)
            sys.stdout.write(f"\r🔴 [{bar}] {int((i/steps)*100)}%")
            sys.stdout.flush()
        print(" 🟢 DONE")

    t = threading.Thread(target=visual_bar)
    t.start()
    sd.wait()
    t.join()

    sf.write(filename, recording, SAMPLE_RATE)
    print(f"💾 Saved.")


# --- RECORDING LOOP ---
try:
    end_index = start_index + NUM_SAMPLES
    for i in range(start_index, end_index):
        current_filename = os.path.join(OUTPUT_DIR, f"{FILE_PREFIX}{i:02d}.wav")
        session_current = i - start_index + 1
        print(f"\nProgress: {session_current}/{NUM_SAMPLES} (Total in folder will be {i+1})")
        record_sample(current_filename)
        
    print(f"\n✅ Session completed! Total files: {len(glob.glob(os.path.join(OUTPUT_DIR, '*.wav')))}")

except KeyboardInterrupt:
    print("\n\n🛑 Session interrupted.")
