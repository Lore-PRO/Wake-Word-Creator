
# 🧠 Wake Word Creator v1.0

Wake Word Creator is a comprehensive suite optimized for Python 3.11, designed to record audio samples, manage a balanced dataset, and train a custom "Wake Word" recognition model (similar to "Hey Siri" or "Alexa").
This project automatically detects and utilizes the best available hardware:
- NVIDIA GPU: full support via CUDA for ultra-fast training
- Apple Silicon: optimized for M-series chips via MPS (Metal Performance Shaders)
- CPU: fallback to multi-threaded CPU execution if no GPU is detected


## ✨ Key Features
- Smart Recording Advisor: guided recording system that suggests the ideal number of samples to maintain a balanced dataset (1:3 Ratio)
- Dynamic Augmentation: during training, each sample is mixed in real-time with ambient noises (Cocktail Party Effect) for extreme robustness
- CUDA/MPS Optimized: leverages GPU cores for ultra-fast training
- Smart CPU Management: automatically detects Performance and Efficiency cores to parallelize data loading without slowing down the system
- Live Testing Dashboard: terminal interface with real-time visual feedback, confidence scores, and cooldown management


## 🛠 Installation
1. **Clone the project and enter the folder:**
    ```bash
    git clone [https://github.com/tuo-username/wake-word-creator.git](https://github.com/Lore-PRO/Wake-Word-Creator.git)
    cd Wake-Word-Creator
    ```

2. **Create and activate the virtual environment (recommended):**
    ```
    python3.11 -m venv venv
    source venv/bin/activate    # On Windows: venv\Scripts\activate
    ```
3. **Install dependencies:**
    ```
    pip install -r requirements.txt
    ```


## 🚀 Workflow

1. Prepare Background Noises (Optional)
If you have raw ambient noise files (MP3, M4A, FLAC) that you want to use for augmentation, place them in the raw_noises/ folder and run:
    ```
    python convert_noises.py
    ```
This utility automatically resamples audio to 16kHz Mono WAV, ensuring perfect compatibility with the training pipeline.

2. Sample Recording
The recording process is structured in a mandatory sequence to ensure perfect dataset balancing:
- step 1: the user records the chosen Wake Word first. This establishes the baseline count (Target: 30-50 samples)
    ```
    python record_positives.py
    ```
- step 2: once the positives are set, the user records "negative" sounds (similar words, distractions, or different voices).
During this phase, the internal Advisor will calculate the ratio in real-time and suggest reaching a 1:3 ratio (3 negatives for every 1 positive) to ensure the model's reliability

    ```
    python record_negatives.py
    ```

3. Training
Launch the neural model training:
    ```
    python train.py
    ```
The system will automatically calculate the Mel spectrogram, apply augmentation, and save the best model in .pth and .onnx formats in the generated_model/folder.

4. Dataset Validation & Stress Test
Before moving to the live test, you can verify the model's robustness against specific noises (rain, office, traffic) at different intensity levels.
    ```
    python validate_dataset.py
    ```
    
5. Real-Time Testing
Verify the model's performance with the microphone:
    ```
    python live_test.py
    ```
Adjust the Sensitivity Threshold (e.g., 0.85) to avoid false positives or missed activations.


🔊 Background Noise Tips

For a robust model, your raw_noises/ folder should contain a diverse "cocktail" of sounds:
Variety is key: Include keyboard typing, dogs barking, door slams, fans, water running, and distant chatter or music.
File Naming: Give your files descriptive names (e.g., loud_traffic.mp3, keyboard_clicks.wav). The validate_dataset.py utility will show these names during stress tests, helping you identify which sounds confuse your model.

Duration:
- minimum: 10 seconds (to allow the trainer to pick different random crops)
- maximum: 5-10 minutes (to avoid memory overhead and extremely large files)
- total volume: aim for at least 5-10 minutes of total unique background noise


## 🏋️ Training Details & Performance

To ensure maximum recognition robustness, the training process generates a new and unique version of the dataset at every single epoch through an "On-the-fly Data Augmentation" technique.
Instead of simply reading static files, the system operates as a real-time DJ Mixer: the CPU dynamically selects between 1 and 3 different background noises and overlays them onto each voice sample using random volumes and temporal crops at each epoch.
The intent of this strategy is to prevent the model from ever hearing the exact same sound twice, forcing it to ignore environmental interference and focus exclusively on the vocal pattern of the wake word.
While this continuous re-mixing requires a significant CPU workload and may noticeably slow down epoch progression on platforms like Google Colab, which offers limited CPU resources (typically 2 cores), the final effect is fundamental:
it prevents the model from memorizing specific noise patterns (overfitting), resulting in a system capable of generalizing and operating with extreme precision even in noisy, real-world environments.
On PCs with multiple cores, the trainer is designed to utilize between 75-80% of available CPU resources, significantly accelerating performance and drastically reducing training time while maintaining maximum model robustness.

Note on performance:
- CPU Intensive: because noise mixing happens in real-time, you will notice significant CPU usage during training
- Google Colab: on free-tier environments like Colab, the limited CPU cores often become a bottleneck.
This might result in slower epoch times compared to a local workstation, but it ensures the model is exposed to millions of unique noise combinations, leading to superior real-world accuracy


## 📂 Project Structure
```text
.
├── dataset/
│   ├── positives/      # Target wake word samples (.wav)
│   ├── negatives/      # Distraction sounds and similar words (.wav)
│   └── noises/         # Converted background noises (.wav)
├── raw_noises/         # Raw background noises (MP3, M4A, .wav, FLAC)
├── generated_model/    # Exported .pth and .onnx models
├── convert_noises.py
├── record_positives.py
├── record_negatives.py
├── train.py
├── validate_dataset.py
├── live_test.py
├── requirements.txt
└── README.md
```


## ⚖️ License

This project is licensed under **CC BY-NC 4.0** (Creative Commons Attribution-NonCommercial 4.0 International).

### What this means:
- **Attribution (BY)**: You must give appropriate credit to the original author and provide a link to this repository.
- **Non-Commercial (NC)**: This software and its derivatives may not be used for commercial purposes. This includes selling the code, integrating it into paid products, or using it to provide commercial services.

### Usage Scope
* **Allowed**: Strictly for individual, private, and hobbyist use.
* **Institutional & Commercial Use**: Use by Academic, Educational, Research institutions, or Commercial & Business entities (including any form of company or corporation) is strictly prohibited without a specific agreement. Please contact the author for licensing inquiries.

📧 Contact: *lorenzo.prometti@gmail.com*
