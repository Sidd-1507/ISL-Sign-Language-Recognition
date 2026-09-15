# Indian Sign Language recognition

Webcam app that reads Indian Sign Language letters and digits (A–Z and 0–9) and types them as text.

A frame from the camera is cropped around the hand with MediaPipe. That crop is classified by a Vision Transformer (ViT-B/16) saved as `signformer_final.pth`. The UI is a Streamlit page with two modes: take one snapshot when your hand is in place, or run continuously from the webcam.

Capture mode is the more reliable of the two. You pose, click, and get a prediction (including top-3 and confidence) instead of fighting a live loop.

## Setup

Python 3.11 is a safe choice. Use a virtualenv.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Put `signformer_final.pth` in this folder. Then:

```bash
streamlit run sign_language_app.py
```

Allow the browser to use the camera.

## Training (optional)

The weights are not in this repo (they are large). To rebuild them:

1. `python download_hf_dataset.py` — download the ISL image set from Hugging Face.
2. `python pipeline_fix_train.py` — crop every training image the same way the webcam crop works, then fine-tune the ViT (classification head first, then the full net).
3. Rename the output checkpoint to `signformer_final.pth` if the app should load it.

`collect_missing_data.py` is for recording extra webcam samples of classes that were weak in the original set.

## Files

| File | Role |
| --- | --- |
| `sign_language_app.py` | Streamlit app |
| `pipeline_fix_train.py` | Recrop dataset + fine-tune |
| `download_hf_dataset.py` | Fetch images from Hugging Face |
| `collect_missing_data.py` | Extra webcam samples |
| `diagnose_pipeline.py` | Check crop / model wiring |
| `test_cam.py` | Quick camera check |
| `requirements.txt` | Python packages |

`dataset/` and `*.pth` are gitignored.
