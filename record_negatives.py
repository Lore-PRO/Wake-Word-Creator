
# -------------------------------------------------------------------------------
# Project: Wake Word Creator
# File: record_negatives.py
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
POS_PATH = os.path.join(BASE_DATASET_PATH, "positives")
OUTPUT_DIR = os.path.join(BASE_DATASET_PATH, "negatives")
FILE_PREFIX = "negative_"


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
print("\n--- 🧠 SMART DATASET ADVISOR ---")

if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

num_positives = len(glob.glob(os.path.join(POS_PATH, "*.wav")))
num_negatives_existing = len(glob.glob(os.path.join(OUTPUT_DIR, "*.wav")))

print(f"📊 Current Status:")
print(f"   👉 Positives: {num_positives}")
print(f"   👉 Negatives: {num_negatives_existing}")


# --- STEP 2: CLEANUP DECISION (Must happen BEFORE diagnosis) ---
start_index = 0
if num_negatives_existing > 0:
    should_clear = input(f"\n🗑  Do you want to clear the existing '{OUTPUT_DIR}' folder and start fresh? (y/N): ").strip().lower()
    
    if should_clear == 'y':
        if os.path.exists(OUTPUT_DIR):
            shutil.rmtree(OUTPUT_DIR)
        os.makedirs(OUTPUT_DIR)
        print(f"🧹 Folder cleared. Resetting count to 0.")
        num_negatives_existing = 0
    else:
        print(f"📂 Keeping existing files.")
        start_index = get_next_index(OUTPUT_DIR, FILE_PREFIX)
else:
    start_index = 0


# --- STEP 3: DIAGNOSIS & TARGETS (Based on post-cleanup status) ---
if num_positives == 0:
    print("\n❌ ERROR: No positive samples found!")
    print("   Please record POSITIVES first. I need them to calculate the ratio.")
    min_target = 0
    ideal_target = 30
    max_limit = 100
else:
    min_target = num_positives * 2      # 1:2
    ideal_target = num_positives * 3    # 1:3
    max_limit = num_positives * 4       # 1:4

    print(f"\n🎯 TARGETS (Based on {num_positives} positives):")
    print(f"   ❌ 1:1 Ratio ({num_positives} negs):  Unsafe (High False Positives)")
    print(f"   ⚠️ 1:2 Ratio ({min_target} negs):  Acceptable (Minimum)")
    print(f"   ✅ 1:3 Ratio ({ideal_target} negs):  Rock-Solid (Recommended)")
    print(f"   🛑 1:4 Ratio ({max_limit} negs):  Paranoid (Do not exceed!)")


# --- STEP 4: INPUT LOGIC ---

# CASE A: TOO MANY NEGATIVES (Safety Block)
if num_positives > 0 and num_negatives_existing >= max_limit:
    print(f"\n🛑 DIAGNOSIS: TOO MANY NEGATIVES.")
    print(f"   You have {num_negatives_existing}, which exceeds the safe limit of {max_limit}.")
    
    force_choice = input("\n   Do you really want to force add more samples? (y/N): ").strip().lower()
    
    if force_choice != 'y':
        print("🛑 Smart choice. Session finished.")
        exit()
    else:
        try:
            user_val = input("   How many samples to add? (Enter number): ")
            NUM_SAMPLES = int(user_val)
        except ValueError:
            print("❌ Invalid input.")
            exit()

# CASE B: NORMAL / NEEDS MORE
else:
    gap_to_ideal = max(0, ideal_target - num_negatives_existing)
    
    if num_negatives_existing < min_target:
        print(f"\n❌ DIAGNOSIS: WEAK BALANCE. You are in the danger zone.")
        suggested_samples = max(10, gap_to_ideal)
        prompt_intro = f"Suggested: {suggested_samples} to reach target"
        
    elif num_negatives_existing < ideal_target:
        print(f"\n⚠️  DIAGNOSIS: ACCEPTABLE (Min 1:2 reached). Recommended: Reach 1:3 for best results.")
        suggested_samples = gap_to_ideal
        prompt_intro = f"Suggested: {suggested_samples} to reach target"
        
    else:
        # GREEN ZONE: Default is now 0 (Skip)
        print(f"\n✅ DIAGNOSIS: ROCK-SOLID BALANCE (Ratio 1:3 reached).")
        suggested_samples = 0 
        prompt_intro = f"Suggested: 0 - You are good to go"
    
    try:
        if suggested_samples == 0:
            prompt_text = f"\n🔢 Enter number of samples to ADD [Press ENTER to Skip]: "
        else:
            prompt_text = f"\n🔢 Enter number of samples to ADD [Press ENTER for Default: {suggested_samples}]: "
            
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

# Final safety check
final_total = num_negatives_existing + NUM_SAMPLES
if max_limit > 0 and final_total > max_limit:
    print(f"\n⚠️  FINAL WARNING: This will result in {final_total} negatives (Ratio > 1:4).")
    confirm = input("   Are you absolutely sure? (y/N): ").strip().lower()
    if confirm != 'y':
        print("🛑 Session aborted.")
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
