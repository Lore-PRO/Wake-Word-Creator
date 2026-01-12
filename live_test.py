
# -------------------------------------------------------------------------------
# Project: Wake Word Creator
# File: live_test.py
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
import numpy as np
import onnxruntime as ort
import librosa
import os
import time
import sys
from mic_config import DEVICE_INDEX, DEVICE_NAME


# --- CONFIGURATION ---
MODEL_PATH = "generated_model/wake_word_model.onnx"
RATE = 16000

# SLIDING WINDOW PARAMETERS
WINDOW_SECONDS = 1.0     
STRIDE_SECONDS = 0.1     
CHUNK = int(RATE * STRIDE_SECONDS) 
WINDOW_SAMPLES = int(RATE * WINDOW_SECONDS)

# TIMING PARAMETERS
COOLDOWN_SECONDS = 2.0   
WARMUP_SECONDS = 2.0     


# --- VISUALIZATION SETTINGS ---
HISTORY_MAX_LINES = 10   
detection_history = []   


# --- USER INPUT FOR SENSITIVITY ---
os.system('cls' if os.name == 'nt' else 'clear')

print("\n--- SETUP ---")
print(f"Detected Input Device Index: {DEVICE_INDEX}")
try:
    user_val = input("📶 Enter sensitivity threshold (0.1 - 0.99) [Default: 0.85]: ")
    THRESHOLD = 0.85 if user_val.strip() == "" else float(user_val)
except ValueError:
    THRESHOLD = 0.85


# --- PREPARE STATIC HEADER ---
DASHBOARD_HEADER = (
    f"\n--- CONFIGURATION ---\n"
    f"🎙  Input Device: [{DEVICE_INDEX}] {DEVICE_NAME}\n"
    f"✅ Sensitivity:  {THRESHOLD}\n"
    f"⚡️ Update Rate:  {STRIDE_SECONDS}s\n"
    f"-----------------------------------------------------------------\n"
)


# --- INITIALIZATION ---
if not os.path.exists(MODEL_PATH):
    print(f"❌ Model {MODEL_PATH} not found!")
    exit()

session = ort.InferenceSession(MODEL_PATH)

def preprocess_buffer(audio_buffer):
    """Preprocesses the float32 audio buffer directly."""
    mel = librosa.feature.melspectrogram(
        y=audio_buffer, sr=RATE, n_mels=40, n_fft=1024, hop_length=512 
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)
    mel_norm = (mel_db + 80) / 80 
    
    if mel_norm.shape[1] > 32:
        mel_norm = mel_norm[:, :32]
    elif mel_norm.shape[1] < 32:
        mel_norm = np.pad(mel_norm, ((0, 0), (0, 32 - mel_norm.shape[1])))
    
    return np.expand_dims(mel_norm.astype(np.float32), axis=0)


# --- AUDIO STREAM SETUP ---
p = pyaudio.PyAudio()
stream = p.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=RATE,
    input=True,
    input_device_index=DEVICE_INDEX,
    frames_per_buffer=CHUNK
)

audio_buffer = np.zeros(WINDOW_SAMPLES, dtype=np.float32)
last_detection_time = 0
start_time = time.time()

os.system('cls' if os.name == 'nt' else 'clear')
print("\033[?25l", end="")

try:
    while True:
        # 1. Read Audio
        data = stream.read(CHUNK, exception_on_overflow=False)
        new_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
        
        # 2. Update Buffer
        audio_buffer = np.roll(audio_buffer, -len(new_audio))
        audio_buffer[-len(new_audio):] = new_audio
        
        # 3. Inference
        input_tensor = preprocess_buffer(audio_buffer)
        outputs = session.run(None, {'input': input_tensor})
        score = outputs[0][0][0]
        
        # 4. Logic & Detection
        current_time = time.time()
        is_warmup = (current_time - start_time < WARMUP_SECONDS)
        is_cooldown = (current_time - last_detection_time < COOLDOWN_SECONDS)
        
        display_score = 0.0 if is_warmup else score

        if score > THRESHOLD and not is_cooldown and not is_warmup:
            timestamp = time.strftime("%H:%M:%S")
            detection_history.append(f"[{timestamp}] 🔥 WAKE WORD DETECTED! (Score: {score:.4f})")
            if len(detection_history) > HISTORY_MAX_LINES:
                detection_history.pop(0)
            
            last_detection_time = current_time
            is_cooldown = True 

        # 5. VISUALIZATION
        
        # A. Reset Cursor to Home (0,0)
        print("\033[H", end="") 
        
        # B. Print the Static Header
        print(DASHBOARD_HEADER, end="")
        
        # C. Print Status Bar
        if is_cooldown:
            print(f"⏳ COOLDOWN | 🔄  Reloading...                  \033[K")
        else:
            bar_width = 30
            filled_len = int(display_score * bar_width)
            bar = "█" * filled_len + "░" * (bar_width - filled_len)
            
            if is_warmup:
                status_text = "🟠 WARMUP  " 
            else:
                status_text = "🟢 LISTENING"
            
            print(f"{status_text} | Score: {display_score:.4f} |{bar}| \033[K")

        print("-" * 65 + "\033[K")

        # D. Print History
        for i in range(HISTORY_MAX_LINES):
            if i < len(detection_history):
                print(f"{detection_history[i]}\033[K")
            else:
                print("\033[K")
        
        # E. Clear bottom garbage
        print("\033[J", end="")

except KeyboardInterrupt:
    print("\033[?25h") 
    print("\n🛑 Stopping...")
finally:
    stream.stop_stream()
    stream.close()
    p.terminate()
