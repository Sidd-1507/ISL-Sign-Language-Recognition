# 🤟 Indian Sign Language (ISL) Recognition App

This project is a real-time **Indian Sign Language (ISL) Recognition System** that translates hand gestures into text using a webcam. It features a complete pipeline from dataset collection and preprocessing to deep learning model fine-tuning and a beautiful, interactive Streamlit frontend.

The model is capable of recognizing all **36 ISL alphanumeric classes** (A-Z, 0-9).

---

## ✨ Features

- **Real-Time Inference:** Uses a webcam to capture hand signs and translates them instantly.
- **Robust Cropping Pipeline:** Uses [MediaPipe](https://developers.google.com/mediapipe) to track hands dynamically. If no hand is found, it falls back to a smart center crop.
- **Deep Learning Model:** Fine-tuned Vision Transformer (ViT) architecture (`signformer`) for high-accuracy gesture classification.
- **Automated Dataset Pipeline:** 
  - Scripts to automatically download the ISL dataset (42,000+ images) from Hugging Face.
  - MediaPipe preprocessing step to crop and prepare the dataset specifically for webcam-style inference.
- **Interactive UI:** A highly polished web app built with [Streamlit](https://streamlit.io/) for easy user interaction.

## 📁 Project Structure

- `sign_language_app.py`: The main Streamlit web application for real-time translation.
- `pipeline_fix_train.py`: The core pipeline script that handles re-cropping the dataset using MediaPipe, and fine-tuning the base model (`signformer_final.pth`).
- `collect_missing_data.py`: A utility script to capture specific missing classes (e.g., '0' or 'V') using your own webcam and MediaPipe validation to ensure the hand is visible.
- `download_hf_dataset.py`: Automatically fetches the ISL dataset from HuggingFace to populate your `dataset/split_dataset` folders.
- `requirements.txt`: Python dependencies required to run the project.
- `.gitignore`: Ensures that large datasets (`dataset/`), heavy model checkpoints (`*.pth`), and virtual environments are not pushed to GitHub.

## 🚀 Getting Started

### 1. Installation

Requires Python 3.9+ (Python 3.11 recommended). It is highly recommended to use a virtual environment.

```bash
# Create and activate a virtual environment
python3 -m venv venv311
source venv311/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Prepare the Dataset and Train (Optional)
If you want to train the model from scratch or fine-tune it with your own data:
1. Run `python download_hf_dataset.py` to get the base data.
2. Run `python pipeline_fix_train.py` to trigger the automated MediaPipe cropping (Step 1) and PyTorch fine-tuning (Step 2).

### 3. Run the Web Application
Make sure you have your trained model (`signformer_final.pth`) in the root directory.

```bash
streamlit run sign_language_app.py
```
This will open the interface in your web browser. Grant webcam permissions and start signing!

## 🧠 Model Architecture & Training

The training script (`pipeline_fix_train.py`) uses a two-phase transfer learning approach on a Vision Transformer (ViT-B/16):
1. **Phase 1 (Head Only):** The base network is frozen, and only the final classification head is trained for stability (high learning rate).
2. **Phase 2 (Full Network):** The entire network is unfrozen and fine-tuned with a lower learning rate to adapt to webcam background noise and specific hand shapes.

## 🛠 Tech Stack

- **Frontend:** Streamlit
- **Machine Learning / Vision:** PyTorch, torchvision
- **Hand Tracking:** Google MediaPipe, OpenCV (`cv2`)
- **Image Processing:** Pillow, Rembg

## 📝 License

This project is open-source and available for educational and research purposes.
