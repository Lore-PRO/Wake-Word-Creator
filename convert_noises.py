
# -------------------------------------------------------------------------------
# Project: Wake Word Creator
# File: convert_noises.py
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
import glob
import librosa
import soundfile as sf


# --- CONFIGURATION ---
INPUT_FOLDER = "raw_noises"      # Source folder for raw audio files
OUTPUT_FOLDER = "dataset/noises" # Destination for processed samples
TARGET_SR = 16000                # Audio is resampled to 16kHz for model consistency

def convert_audio_files():
    """
    Scans the input folder and converts supported audio formats to 
    16kHz Mono WAV files for the noise dataset.
    """
    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)
    
    extensions = ('*.mp3', '*.wav', '*.m4a', '*.flac')
    files_to_convert = []
    for ext in extensions:
        files_to_convert.extend(glob.glob(os.path.join(INPUT_FOLDER, ext)))

    if not files_to_convert:
        print(f"⚠️  No audio files found in '{INPUT_FOLDER}'.")
        print("   Please put your MP3/M4A files there.")
        return

    print(f"🚀 Found {len(files_to_convert)} files. Starting conversion to 16kHz Mono WAV...")

    for file_path in files_to_convert:
        filename = os.path.basename(file_path)
        name_no_ext = os.path.splitext(filename)[0]
        output_path = os.path.join(OUTPUT_FOLDER, f"{name_no_ext}.wav")

        print(f"⏳ Processing: {filename}...", end="\r")

        try:
            # librosa handles resampling and mono conversion during loading
            # sr=TARGET_SR ensures 16000Hz, mono=True is the default behavior
            audio, sr = librosa.load(file_path, sr=TARGET_SR, mono=True)

            sf.write(output_path, audio, TARGET_SR)
            print(f"✅ Converted: {filename} -> {os.path.basename(output_path)}")

        except Exception as e:
            print(f"❌ Error converting {filename}: {e}")

    print("\n✨ Conversion complete! Your noises are ready in 'dataset/noises/'.")

if __name__ == "__main__":
    convert_audio_files()
