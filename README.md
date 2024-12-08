# ATC-Models: Speech-to-Text and Intent Classification for Air Traffic Control Communications

## Overview

This project aims to enhance Air Traffic Control (ATC) communication safety and efficiency by developing AI models for speech-to-text (STT) transcription and intent classification. ATC communication often involves challenges such as static noise, accent variation, and complex aviation-specific terminology. Our models are fine-tuned to handle these nuances, providing a foundation for safer and more effective real-time decision-making.

## Project Goals

1. **Speech-to-Text (STT):** 
   - Transcribe ATC audio accurately, even in noisy environments or with varying accents.
   - Fine-tune existing STT models like Wav2Vec 2.0, DeepSpeech, Whisper, and Vosk for aviation-specific language.

2. **Speech-to-Intent Classification:**
   - Identify and classify intents in transcribed ATC communications (e.g., clearance requests, alerts).
   - Utilize NLP techniques to train intent classification models specific to ATC scenarios.

3. **Stretch Goal:**
   - Integrate real-time data (e.g., weather, trajectory) to suggest or automate actions based on detected intents.

## Dataset and Tools

### Datasets
- **Audio Dataset:**
  - TartanAviation dataset, downloaded per instructions from [TartanAviation README](https://github.com/castacks/TartanAviation/blob/main/audio/README.md).
- **Weather Data:**
  - Individual weather statistics (BTP and AGC files) for integrating multimodal data.

### Tools and Libraries
- **Noise Reduction:** Librosa
- **Speech-to-Text Models:** Wav2Vec 2.0, DeepSpeech, Whisper, Vosk
- **Programming Languages:** Python
- **Frameworks:** PyTorch, Hugging Face Transformers

## Project Workflow

### 1. Speech-to-Text (STT) Pipeline
- **Goal:** Accurately transcribe ATC audio.
- **Methodology:**
  - Preprocess audio using noise-reduction techniques.
  - Fine-tune models using aviation-specific language data.
- **Output:** Clean, labeled transcriptions of ATC communications.

### 2. Intent Classification Pipeline
- **Goal:** Detect intents behind transcribed messages.
- **Methodology:**
  - Train NLP models with transcriptions labeled for intents like clearance requests, warnings, and alerts.
- **Output:** Classified intent categories for ATC commands.

### 3. Stretch Goal: Multimodal Integration
- **Objective:** Suggest or automate actions by combining intent data with weather statistics and trajectory data.
- **Methodology:** Integrate intent detection with real-time weather data.

## Contributions

- **Michael De Leon:** 
  - Data preparation and quality assurance, including downloading and preprocessing audio datasets, and implementing noise-reduction techniques.
- **Tadhbir Singh:** 
  - Fine-tuning speech recognition models for ATC language, testing model accuracy, and setting up cloud-based training environments.

## Installation and Usage

1. Clone the repository:
   ```
   git clone https://github.com/GitAIwithMike/ATC-Models.git
   ```
2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Run the preprocessing script to prepare datasets:
   ```
   python preprocess.py
   ```
4. Fine-tune the STT models:
   ```
   python train_stt.py --model <model_name>
   ```
5. Run intent classification training:
   ```
   python train_intent.py
   ```

## Future Work
- Extend the project to integrate trajectory prediction and real-time decision support.
- Further refine models to handle diverse accents and dialects in ATC communication.
- Test models in simulated ATC environments for real-world validation.

## References

1. Patrikar, J., Dantas, J., Moon, B., Hamidi, M., Ghosh, S., Keetha, N., Higgins, I., Chandak, A., Yoneyama, T., & Scherer, S. (2024). **TartanAviation: Image, Speech, and ADS-B Trajectory Datasets for Terminal Airspace Operations**. *arXiv preprint arXiv:2403.03372*. Retrieved from [https://arxiv.org/pdf/2403.03372.pdf](https://arxiv.org/pdf/2403.03372.pdf)
