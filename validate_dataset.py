
# -------------------------------------------------------------------------------
# Project: Wake Word Creator
# File: validate_dataset.py
# Author: Lorenzo Prometti (@Lore-PRO)
# License: CC BY-NC 4.0 (Attribution-NonCommercial 4.0 International)
#
# This work is licensed under the Creative Commons Attribution-NonCommercial 
# 4.0 International License. To view a copy of this license, visit 
# http://creativecommons.org/licenses/by-nc/4.0/
#
# For commercial licensing inquiries, please contact: lorenzo.prometti@gmail.com
# -------------------------------------------------------------------------------


import onnxruntime as ort
import librosa
import numpy as np
import glob
import os
import random


# --- CONFIGURATION ---
DEFAULT_THRESHOLD = 0.85
MODEL_FOLDER = "generated_model"
DATASET_FOLDER = "dataset/positives" 
NOISE_FOLDER = "dataset/noises"
NOISE_LEVELS = [0.0, 0.1, 0.25, 0.5] 


# --- HELPER FUNCTIONS ---
def configure_threshold():
    print(f"\n--- 📶 SENSITIVITY SETTINGS ---")
    print(f"Standard values: 0.5 (Permissive) | 0.85 (Balanced) | 0.95 (Strict)")
    
    user_input = input(f"👉 Enter Threshold (Press Enter for Default {DEFAULT_THRESHOLD}): ").strip()
    
    if not user_input:
        print(f"✅ Using default: {DEFAULT_THRESHOLD}")
        return DEFAULT_THRESHOLD
    
    try:
        val = float(user_input)
        if 0.0 < val < 1.0:
            print(f"✅ Custom threshold set: {val}")
            return val
        else:
            print(f"⚠️  Value out of range (0-1). Reverting to default.")
            return DEFAULT_THRESHOLD
    except ValueError:
        print(f"⚠️  Invalid input. Reverting to default.")
        return DEFAULT_THRESHOLD

def select_noise_file(target_length):
    """Lists available noise files and returns a randomly cropped chunk."""
    noise_files = sorted(glob.glob(os.path.join(NOISE_FOLDER, "*.wav")))
    
    if not noise_files:
        print("⚠️  No noise files found. Using silence.")
        return np.zeros(target_length), "None"

    print(f"\n--- 🔊 NOISE SELECTION ({len(noise_files)}) ---")
    
    half = (len(noise_files) + 1) // 2
    for i in range(half):
        idx_left = i
        str_left = f"[{idx_left+1}] {os.path.basename(noise_files[idx_left])}"
        
        idx_right = i + half
        str_right = ""
        if idx_right < len(noise_files):
            str_right = f"[{idx_right+1}] {os.path.basename(noise_files[idx_right])}"
            
        print(f"{str_left:<40} {str_right}")
    
    choice = input("\n👉 Select noise number (Press Enter for Random): ").strip()
    
    if choice.isdigit():
        idx = int(choice) - 1
        if 0 <= idx < len(noise_files):
            selected_path = noise_files[idx]
            print(f"✅ Selected: {os.path.basename(selected_path)}")
        else:
            print("⚠️  Invalid number. Defaulting to Random.")
            selected_path = random.choice(noise_files)
    else:
        print("🎲 Random selection chosen.")
        selected_path = random.choice(noise_files)

    n_src, _ = librosa.load(selected_path, sr=16000)
    
    if len(n_src) < target_length:
        n_src = np.pad(n_src, (0, target_length - len(n_src)))
    
    start = random.randint(0, len(n_src) - target_length)
    return n_src[start : start + target_length], os.path.basename(selected_path)


def mix_specific_noise(clean_audio, noise_chunk, noise_factor):
    """Mixes noise into audio using RMS normalization to maintain relative levels."""
    if noise_factor == 0.0: return clean_audio
    
    clean_rms = np.sqrt(np.mean(clean_audio**2))
    noise_rms = np.sqrt(np.mean(noise_chunk**2))
    
    if noise_rms < 1e-9: return clean_audio

    normalized_noise = noise_chunk * (clean_rms / noise_rms)
    noisy_audio = clean_audio + (normalized_noise * noise_factor)
    
    max_val = np.max(np.abs(noisy_audio))
    if max_val > 0: noisy_audio = noisy_audio / max_val
    
    return noisy_audio

def preprocess_to_tensor(y, sr=16000):
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, n_fft=1024, hop_length=512)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80) / 80 

    if mel_norm.shape[1] > 32:
        mel_norm = mel_norm[:, :32]
    elif mel_norm.shape[1] < 32:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, 32 - mel_norm.shape[1])))

    return np.expand_dims(mel_norm.astype(np.float32), axis=0)


# --- EXECUTION ---
files = glob.glob(os.path.join(DATASET_FOLDER, "*.wav"))
if not files: files = glob.glob("positive_*.wav")
if not files: raise FileNotFoundError("❌ Error: No wav files found for analysis!")

filename = random.choice(files)
print(f"📂 Analyzing Target: {os.path.basename(filename)}")

y_base, sr = librosa.load(filename, sr=16000, duration=1.0)
if len(y_base) < 16000: y_base = np.pad(y_base, (0, 16000 - len(y_base)))
else: y_base = y_base[:16000]

model_path = os.path.join(MODEL_FOLDER, "wake_word_model.onnx")
if not os.path.exists(model_path): raise FileNotFoundError(f"❌ Error: Model not found at {model_path}")
session = ort.InferenceSession(model_path)

noise_chunk, noise_name = select_noise_file(len(y_base))
current_threshold = configure_threshold()

print(f"\n⚡️ STRESS TEST (Threshold: {current_threshold})")
print(f"🔊 Noise Source: {noise_name}") 
print("-" * 65)
print(f"{'DIFFICULTY':<15} | {'NOISE RATIO':<12} | {'SCORE':<10} | {'RESULT'}")
print("-" * 65)

for noise_factor in NOISE_LEVELS:
    y_dirty = mix_specific_noise(y_base, noise_chunk, noise_factor)
    
    input_tensor = preprocess_to_tensor(y_dirty)
    outputs = session.run(None, {'input': input_tensor})
    score = outputs[0][0][0]
    
    status = "✅ DETECTED" if score > current_threshold else "❌ MISSED"
    
    if noise_factor == 0.0: level_desc = "Clean"
    elif noise_factor <= 0.1: level_desc = "Easy"
    elif noise_factor <= 0.25: level_desc = "Medium"
    else: level_desc = "Hard"
        
    print(f"{level_desc:<15} | {noise_factor:<12} | {score:.4f}     | {status}")

print("-" * 65)
