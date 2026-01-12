
# -------------------------------------------------------------------------------
# Project: Wake Word Creator
# File: train.py
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
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import librosa
import numpy as np
import torch.onnx
import random
import multiprocessing

# --- CLASSES (Must be outside main) ---
class WakeWordDataset(Dataset):
    def __init__(self, pos_path, neg_path, noise_path):
        self.samples = [] 
        self.noises = []
        
        print("⏳ Loading dataset into RAM...")

        noise_files = glob.glob(os.path.join(noise_path, "*.wav"))
        for f in noise_files:
            y, _ = librosa.load(f, sr=16000)
            self.noises.append(y)

        for f in glob.glob(os.path.join(pos_path, "*.wav")):
            y, _ = librosa.load(f, sr=16000, duration=1.0)
            if len(y) < 16000: y = np.pad(y, (0, 16000 - len(y)))
            else: y = y[:16000]
            
            self.samples.append({'audio': y, 'label': 1, 'augment': False})
            
            for _ in range(3): 
                self.samples.append({'audio': y, 'label': 1, 'augment': True})

        for f in glob.glob(os.path.join(neg_path, "*.wav")):
            y, _ = librosa.load(f, sr=16000, duration=1.0)
            if len(y) < 16000: y = np.pad(y, (0, 16000 - len(y)))
            else: y = y[:16000]
            
            self.samples.append({'audio': y, 'label': 0, 'augment': False})
            
            for _ in range(3):
                self.samples.append({'audio': y, 'label': 0, 'augment': True})

        # Generate placeholders for background-only noise
        for _ in range(100):
            self.samples.append({'audio': None, 'label': 0, 'augment': True, 'pure_noise': True})

        print(f"✅ Loaded {len(self.samples)} samples ready for dynamic processing.")

    def __len__(self): 
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        y = item['audio']
        label = item['label']
        
        if item.get('pure_noise'):
            y = np.zeros(16000, dtype=np.float32)
            y = self._add_dynamic_noise_mix(y, force_noise=True)
            
        elif item['augment']:
            y = y.copy()
            y = self._add_dynamic_noise_mix(y)

        mel_tensor = self._to_mel(y)
        return mel_tensor, torch.tensor(label).float()

    def _add_dynamic_noise_mix(self, y, force_noise=False):
        """
        DYNAMIC NOISE INJECTION (DJ MIXER / COCKTAIL PARTY EFFECT)
        
        To prevent overfitting and ensure high robustness, noise is not 
        pre-recorded but mixed dynamically at each epoch. This approach 
        utilizes CPU resources during training to overlay 1 to 3 random noise 
        layers with varying amplitudes onto the original voice sample.
        
        By generating these combinations on-the-fly, the model is forced to 
        learn the core features of the wake word rather than memorizing 
        specific noise patterns, resulting in significantly better 
        generalization in unpredictable real-world environments.
        """
        if not self.noises: return y
        
        if force_noise: noise_amp = random.uniform(0.5, 1.0)
        else: noise_amp = random.uniform(0.05, 0.20)
        
        num_layers = random.randint(1, 3)
        combined_noise = np.zeros(16000)
        
        for _ in range(num_layers):
            noise_src = random.choice(self.noises)

            if len(noise_src) < 16000:
                noise_chunk = np.pad(noise_src, (0, 16000 - len(noise_src)))
            else:
                start = random.randint(0, len(noise_src) - 16000)
                noise_chunk = noise_src[start:start+16000]
            
            combined_noise += noise_chunk * random.uniform(0.5, 1.0)
        
        max_val = np.max(np.abs(combined_noise))
        if max_val > 0: combined_noise /= max_val
        
        y_augmented = y + (combined_noise * noise_amp)
        return np.clip(y_augmented, -1.0, 1.0)

    def _to_mel(self, y):
        # Convert audio to Mel Spectrogram (40 mels, 32 time steps)
        mel = librosa.feature.melspectrogram(y=y, sr=16000, n_mels=40, n_fft=1024, hop_length=512)
        mel_db = librosa.power_to_db(mel, ref=np.max)
        mel_norm = (mel_db + 80) / 80 
        
        if mel_norm.shape[1] > 32: mel_norm = mel_norm[:, :32]
        elif mel_norm.shape[1] < 32: mel_norm = np.pad(mel_norm, ((0, 0), (0, 32 - mel_norm.shape[1])))
            
        return torch.tensor(mel_norm).float().unsqueeze(0)

class WakeWordModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.network = nn.Sequential(
            nn.Linear(40 * 32, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.network(self.flatten(x))


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    multiprocessing.freeze_support()
    
    # Device selection (CUDA / MPS / CPU)
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("🚀 Nvidia found: using GPU (CUDA)")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
        print("🚀 Apple Silicon found: using Metal Performance Shaders (MPS)")
    else:
        device = torch.device("cpu")
        print("💻 No accelerator found: using CPU")

    # SMART WORKER CALCULATION
    # The hardware environment is detected to optimize data loading strategies.
    total_cores = os.cpu_count()
    if total_cores:
        # Case A: Low-resource environments (e.g., Google Colab, Docker containers).
        # All available cores are used to prevent GPU starvation.
        if total_cores <= 2:
            optimal_workers = total_cores 
            print(f"⚠️  Low CPU count detected ({total_cores}). Using ALL cores for data loading.")

        # Case B: High-performance workstations.
        # Approximately 25% of cores are reserved for OS stability, capped at 8 
        # to prevent multiprocessing overhead on macOS (spawn method).
        else:
            # On powerful PCs, leave some breathing room for the OS
            optimal_workers = min(8, max(1, int(total_cores * 0.75)))
            
        print(f"🧠 Smart CPU Management: Allocating {optimal_workers}/{total_cores} workers.")
    else:
        optimal_workers = 2
        print("⚠️ Could not detect CPU count. Defaulting to 2 workers.")

    # Pin Memory Optimization
    # This is set to True for Nvidia GPUs (CUDA) to accelerate data transfer, 
    # while it remains False for Apple Silicon (MPS) to ensure compatibility.
    use_pin_memory = True if device.type == 'cuda' else False

    # Configuration
    base_path = "dataset"
    output_dir = "generated_model"
    pos_path = os.path.join(base_path, "positives")
    neg_path = os.path.join(base_path, "negatives")
    noise_path = os.path.join(base_path, "noises")
    EPOCH = 500

    if not os.path.exists(pos_path):
        print(f"❌ Error: Dataset not found at {base_path}.")
        exit()

    os.makedirs(output_dir, exist_ok=True)


    # Dataset Health Check
    print("\n🔍 Checking dataset balance...")
    pos_files = glob.glob(os.path.join(pos_path, "*.wav"))
    neg_files = glob.glob(os.path.join(neg_path, "*.wav"))
    num_positives = len(pos_files)
    num_negatives = len(neg_files)

    print(f"   👉 Positives found: {num_positives}")
    print(f"   👉 Negatives found: {num_negatives}")

    if num_positives == 0:
        print("❌ ERROR: No positive samples found!")
        exit()

    if num_negatives < num_positives:
        print("\n⛔ CRITICAL ERROR: NOT ENOUGH NEGATIVES.")
        print(f"   Rule: Negatives must be >= Positives.")
        exit()

    min_negatives = num_positives * 2
    ideal_negatives = num_positives * 3
    max_negatives = num_positives * 4

    if num_negatives < min_negatives:
        print(f"\n⚠️  WARNING: WEAK DATASET BALANCE.")
        print(f"   Current Ratio is low ({num_negatives} negs vs {num_positives} pos).")
        print(f"   To prevent False Positives, the model needs to see 'NO' more often than 'YES'.")
        print(f"   🎯 TARGETS:")
        print(f"     - Minimum (1:2): {min_negatives} negatives (Acceptable)")
        print(f"     - Ideal   (1:3): {ideal_negatives} negatives (Rock-Solid)")
        print(f"     - Maximum (1:4): {max_negatives} negatives (Paranoid: do not exceed this!)")
        
        choice = input("   Do you want to continue with a weak dataset? (y/n): ").strip().lower()
        if choice != 'y':
            print("🛑 Training aborted.")
            exit()

    elif num_negatives > max_negatives:
        print(f"\n⚠️  WARNING: TOO MANY NEGATIVES.")
        print(f"   You have {num_negatives} negatives (Ratio > 1:4).")
        print(f"   Recommendation: Keep negatives below {max_negatives} or record more positives.")
        
        choice = input("   Do you want to continue anyway? (y/n): ").strip().lower()
        if choice != 'y':
            print("🛑 Training aborted.")
            exit()

    print("✅ Dataset balance checks out. Proceeding...\n")


    # Training Setup
    dataset = WakeWordDataset(pos_path, neg_path, noise_path)
    
    loader = DataLoader(
        dataset, 
        batch_size=32,
        shuffle=True, 
        num_workers=optimal_workers,
        persistent_workers=True,
        pin_memory=use_pin_memory,
        prefetch_factor=2
    )

    model = WakeWordModel().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    best_loss = float('inf')
    best_weights = None
    best_epoch = 0
    best_accuracy = 0

    print(f"▶️  Starting training on {len(dataset)} samples (dynamic)...")

    # Training Loop
    for epoch in range(EPOCH):
        loss_accum = 0
        correct = 0
        total = 0
        
        model.train()
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            pred = model(x).squeeze()
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            
            loss_accum += loss.item()
            
            predicted_labels = (pred > 0.5).float()
            correct += (predicted_labels == y).sum().item()
            total += y.size(0)
        
        avg_loss = loss_accum / len(loader)
        accuracy = correct / total * 100
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_accuracy = accuracy
            best_weights = model.state_dict()
            best_epoch = epoch + 1

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{EPOCH} - Loss: {avg_loss:.4f} - Accuracy: {accuracy:.2f}%")

    # Export to PTH and ONNX
    print("\n📦 Exporting best model...")

    if best_weights:
        model.load_state_dict(best_weights)
        print(f"🏆 Selected weights from epoch {best_epoch} with Loss: {best_loss:.4f} and Accuracy: {best_accuracy:.2f}%")
    else:
        print("⚠️  Warning: No best weights found.")

    pth_filename = os.path.join(output_dir, "wake_word_weights.pth")
    torch.save(model.state_dict(), pth_filename)
    print(f"   - Weights saved: {pth_filename}")

    model.eval()
    model.to("cpu") 

    dummy_input = torch.randn(1, 40, 32)
    output_filename = os.path.join(output_dir, "wake_word_model.onnx")

    torch.onnx.export(model, dummy_input, output_filename, input_names=['input'], output_names=['output'])
    print(f"   - ONNX model saved: {output_filename}")
