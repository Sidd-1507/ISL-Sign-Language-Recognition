"""
Real-Time ISL Sign Language Translator — Streamlit App (Capture Mode)

Two modes:
  - Capture Mode (default, recommended): take one snapshot when you're
    ready, and the model predicts on that single frame. Much more reliable
    than continuous prediction, since you can position your hand properly
    before capturing, and you can see exactly what the model thought
    (top-3 predictions with confidence) instead of guessing why it's wrong.
  - Continuous Mode: the original live webcam loop, kept as an option.

RUN LOCALLY with:  streamlit run sign_language_app.py
"""

import time
import os
from collections import deque, Counter

import cv2
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from spellchecker import SpellChecker

# ----------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------
MODEL_PATH = "signformer_final.pth"   # or vit_pretrained_final.pth — whichever you're deploying
IMG_SIZE = 224
STABLE_FRAMES = 8
CONF_THRESHOLD = 0.75
AUTO_SPACE_SECONDS = 2.5
SMOOTHING_WINDOW = 10


# ----------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------
@st.cache_resource
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        ckpt = torch.load(MODEL_PATH, map_location=device)
        classes = ckpt["class_names"]
        model = models.vit_b_16(weights=None)
        model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))
        model.load_state_dict(ckpt["model_state_dict"])
        model = model.to(device).eval()
        tf = transforms.Compose([
            transforms.Resize((ckpt["image_size"], ckpt["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(mean=ckpt["mean"], std=ckpt["std"]),
        ])
    except FileNotFoundError:
        classes = [chr(i) for i in range(65, 91)]
        model = None
        tf = None
        device = "cpu"
    return model, classes, tf, device


@st.cache_resource
def load_hand_detector():
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    return mp_hands.Hands(
        static_image_mode=True,          # True: treats each capture as an independent image
        max_num_hands=1,
        min_detection_confidence=0.5,
    )


@st.cache_resource
def load_spellchecker():
    return SpellChecker()


@st.cache_resource
def load_tts_engine():
    import pyttsx3
    return pyttsx3.init()


# ----------------------------------------------------------------------
# HAND CROP + PREDICTION
# ----------------------------------------------------------------------
def crop_hand(frame_rgb, hand_landmarks, margin_ratio=0.20):
    """
    Crops around the hand using a margin PROPORTIONAL to hand size.
    Increased default margin_ratio to 0.20 (from 0.15) to avoid clipping
    fingertips on signs like B, E, F, J where fingers reach near the edge.

    Also returns whether the crop was clipped by the frame edge, so the
    caller can warn the user to step back from the camera.
    """
    h, w, _ = frame_rgb.shape
    xs = [lm.x * w for lm in hand_landmarks.landmark]
    ys = [lm.y * h for lm in hand_landmarks.landmark]

    hand_width  = max(xs) - min(xs)
    hand_height = max(ys) - min(ys)
    margin_x    = hand_width  * margin_ratio
    margin_y    = hand_height * margin_ratio

    x_min_raw, x_max_raw = min(xs) - margin_x, max(xs) + margin_x
    y_min_raw, y_max_raw = min(ys) - margin_y, max(ys) + margin_y

    x_min, y_min = max(0, int(x_min_raw)), max(0, int(y_min_raw))
    x_max, y_max = min(w, int(x_max_raw)), min(h, int(y_max_raw))

    if x_max <= x_min or y_max <= y_min:
        return None, False

    was_clipped = (x_min_raw < 0 or y_min_raw < 0 or x_max_raw > w or y_max_raw > h)
    return frame_rgb[y_min:y_max, x_min:x_max], was_clipped


def smart_center_crop(frame_rgb, crop_fraction=0.65):
    """
    Fallback when MediaPipe cannot detect a hand (e.g. signs A, K, Q, S, X
    that use two hands, unusual poses, or tight fists that confuse MediaPipe).

    Instead of dropping the frame, we take a centred square crop of
    `crop_fraction` of the shorter image dimension and feed that to the model.
    This keeps the model active even when hand detection fails, while still
    removing most of the irrelevant border.

    Returns (crop_rgb, was_clipped=False) — same signature as crop_hand().
    """
    h, w = frame_rgb.shape[:2]
    side   = int(min(h, w) * crop_fraction)
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2)
    y0 = max(0, cy - side // 2)
    x1 = min(w, x0 + side)
    y1 = min(h, y0 + side)
    return frame_rgb[y0:y1, x0:x1], False


@st.cache_resource
def load_bg_remover():
    from rembg import remove, new_session
    session = new_session("u2net")   # good general-purpose foreground/background model
    return remove, session


def remove_background_to_plain(pil_image, session_tuple, bg_color=(255, 255, 255)):
    """
    Strips the background from a hand image and composites the hand onto a
    plain background, matching the training data's plain-background style.
    This directly targets the two problems seen so far: patterned walls/
    ceilings, and faces accidentally included in the crop — both get
    removed since only the segmented foreground (the hand) survives.
    """
    remove_fn, session = session_tuple
    # rembg wants raw bytes or a PIL image; returns an RGBA image with alpha=0
    # wherever it decided is background.
    result_rgba = remove_fn(pil_image, session=session)

    # Composite onto a solid background color using the alpha mask, so no
    # transparent/black edges leak into the model's input.
    background = Image.new("RGB", result_rgba.size, bg_color)
    background.paste(result_rgba, mask=result_rgba.split()[3])  # alpha channel as mask
    return background


def predict_with_topk(model, tf, device, classes, hand_crop_rgb, k=3):
    """Returns list of (class_name, confidence) sorted by confidence, top-k."""
    if model is None or tf is None:
        return [(c, 1.0 / len(classes)) for c in classes[:k]]
    img = Image.fromarray(hand_crop_rgb)
    x = tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = model(x)
        probs = torch.softmax(outputs, dim=1)[0]
    top_probs, top_idxs = probs.topk(k)
    return [(classes[i.item()], p.item()) for p, i in zip(top_probs, top_idxs)]


# ----------------------------------------------------------------------
# STREAMLIT APP STATE
# ----------------------------------------------------------------------
def init_state():
    defaults = {
        "recent_preds": deque(maxlen=SMOOTHING_WINDOW),
        "last_committed_char": None,
        "current_word": "",
        "sentence": "",
        "last_hand_seen_time": time.time(),
        "camera_on": False,
        "last_capture_predictions": None,   # top-k list from the most recent capture
        "last_capture_image": None,         # the cropped hand image, for display
        "last_capture_used_fallback": False, # True when center-crop fallback was used
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def finalize_word():
    word = st.session_state.current_word.strip()
    if not word:
        return
    spell = load_spellchecker()
    corrected = spell.correction(word.lower())
    final_word = (corrected or word).upper()
    st.session_state.sentence = (st.session_state.sentence + " " + final_word).strip()
    st.session_state.current_word = ""
    st.session_state.last_committed_char = None
    st.session_state.recent_preds.clear()


def clear_all():
    st.session_state.current_word = ""
    st.session_state.sentence = ""
    st.session_state.last_committed_char = None
    st.session_state.recent_preds.clear()
    st.session_state.last_capture_predictions = None
    st.session_state.last_capture_image = None
    st.session_state.last_capture_used_fallback = False


def speak(text, mode="Mac Native (say)"):
    if not text.strip():
        return
    if mode == "Mac Native (say)":
        os.system(f'say "{text}" &')
    else:
        engine = load_tts_engine()
        engine.say(text)
        engine.runAndWait()


# ----------------------------------------------------------------------
# CAPTURE MODE — take one photo, predict once, show top-3, let user confirm
# ----------------------------------------------------------------------
def run_capture_mode(model, tf, device, classes, hands):
    st.markdown("### 📸 Capture Mode")
    st.caption(
        "Position your hand clearly in frame, then click below to take a picture and predict. "
        "Background removal (toggle in sidebar) automatically strips walls, faces, and clutter, "
        "compositing your hand onto a plain background to match the training data."
    )

    img_file = st.camera_input("Take a picture of your hand sign")

    if img_file is not None:
        image = Image.open(img_file).convert("RGB")
        frame_rgb = np.array(image)

        results = hands.process(frame_rgb)
        used_fallback = False

        if not results.multi_hand_landmarks:
            # ── SMART FALLBACK ────────────────────────────────────────────────
            # Some ISL signs (A, K, Q, S, X) use two hands, tight fists, or
            # unusual orientations that confuse MediaPipe. Instead of giving up,
            # we use a centred square crop of the frame and still run the model.
            # A sidebar toggle lets the user disable this if they prefer strict
            # hand-detection-only mode.
            if st.session_state.get("use_fallback_crop", True):
                st.info(
                    "🔍 No hand detected by MediaPipe — using smart center-crop as fallback. "
                    "This helps with signs like **A, K, Q, S, X** that use two hands or unusual poses. "
                    "You can disable this in the sidebar."
                )
                hand_crop, _ = smart_center_crop(frame_rgb)
                used_fallback = True
            else:
                st.warning(
                    "No hand detected. Try better lighting, a plainer background, "
                    "or enable 'Smart fallback crop' in the sidebar."
                )
                st.session_state.last_capture_predictions = None
                st.session_state.last_capture_image = None
                hand_crop = None

        else:
            hand_landmarks = results.multi_hand_landmarks[0]
            hand_crop, was_clipped = crop_hand(frame_rgb, hand_landmarks)
            if hand_crop is None:
                st.warning("Hand detected but crop failed — try repositioning and capture again.")
            elif was_clipped:
                st.warning(
                    "⚠️ Your hand looks too close to the camera — part of it may be "
                    "cut off in the crop below. Try stepping back so your whole hand "
                    "fits comfortably in frame before capturing again."
                )

        if hand_crop is not None:
            # Strip the background and composite onto plain white
            if st.session_state.get("use_bg_removal", True):
                try:
                    bg_remover = load_bg_remover()
                    hand_crop_pil = Image.fromarray(hand_crop)
                    cleaned_pil = remove_background_to_plain(hand_crop_pil, bg_remover)
                    hand_crop_for_model = np.array(cleaned_pil)
                except Exception as e:
                    st.warning(f"Background removal failed ({e}), using raw crop instead.")
                    hand_crop_for_model = hand_crop
            else:
                hand_crop_for_model = hand_crop

            predictions = predict_with_topk(model, tf, device, classes, hand_crop_for_model, k=3)
            st.session_state.last_capture_predictions = predictions
            st.session_state.last_capture_image = hand_crop_for_model
            st.session_state.last_capture_used_fallback = used_fallback

    # --- Show results of the most recent capture, if any ---
    if st.session_state.last_capture_predictions:
        col_img, col_preds = st.columns([1, 1])

        with col_img:
            st.markdown("**Cropped hand region sent to the model:**")
            st.image(st.session_state.last_capture_image, width=250)

        with col_preds:
            st.markdown("**Top 3 predictions:**")
            for i, (cls, conf) in enumerate(st.session_state.last_capture_predictions):
                bar_label = f"{'🥇' if i == 0 else '🥈' if i == 1 else '🥉'} {cls}: {conf*100:.1f}%"
                st.progress(min(conf, 1.0), text=bar_label)

            top_class, top_conf = st.session_state.last_capture_predictions[0]

            st.markdown("---")
            c1, c2 = st.columns(2)
            with c1:
                if st.button(f"✅ Accept '{top_class}' and add to word"):
                    st.session_state.current_word += top_class
                    st.session_state.last_capture_predictions = None
                    st.rerun()
            with c2:
                manual_letter = st.selectbox("Or pick manually:", options=classes, key="manual_pick")
                if st.button(f"➕ Add '{manual_letter}' instead"):
                    st.session_state.current_word += manual_letter
                    st.session_state.last_capture_predictions = None
                    st.rerun()


# ----------------------------------------------------------------------
# CONTINUOUS MODE — original live webcam loop (kept as an option)
# ----------------------------------------------------------------------
def process_prediction_continuous(letter, confidence):
    if confidence >= CONF_THRESHOLD:
        st.session_state.recent_preds.append(letter)
        st.session_state.last_hand_seen_time = time.time()
    else:
        st.session_state.recent_preds.append(None)

    votes = [p for p in st.session_state.recent_preds if p is not None]
    if len(votes) < STABLE_FRAMES:
        return

    majority_letter, count = Counter(votes).most_common(1)[0]
    if count < STABLE_FRAMES:
        return

    if majority_letter != st.session_state.last_committed_char:
        st.session_state.current_word += majority_letter
        st.session_state.last_committed_char = majority_letter
        st.session_state.recent_preds.clear()


def check_auto_word_break():
    if (st.session_state.current_word
            and time.time() - st.session_state.last_hand_seen_time > AUTO_SPACE_SECONDS):
        finalize_word()


def run_continuous_mode(model, tf, device, classes, hands, cam_index):
    frame_window = st.image([])
    status_placeholder = st.empty()

    cap = cv2.VideoCapture(cam_index)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        status_placeholder.error(
            f"Could not open camera index {cam_index}. Try a different index in the sidebar."
        )
        return

    while st.session_state.camera_on:
        ret, frame = cap.read()
        if not ret:
            status_placeholder.error("Failed to read from camera.")
            break

        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(frame_rgb)

        letter_detected = None
        confidence_detected = 0.0

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                import mediapipe as mp
                mp_drawing = mp.solutions.drawing_utils
                mp_hands = mp.solutions.hands
                mp_drawing.draw_landmarks(frame_rgb, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                cropped_result, was_clipped = crop_hand(frame_rgb, hand_landmarks)
                if cropped_result is not None:
                    hand_crop = cropped_result
                    topk = predict_with_topk(model, tf, device, classes, hand_crop, k=1)
                    letter_detected, confidence_detected = topk[0]
                    label_text = f"{letter_detected} ({confidence_detected:.2f})"
                    if was_clipped:
                        label_text += " [too close!]"
                    cv2.putText(
                        frame_rgb, label_text,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                    )
            status_placeholder.empty()
        else:
            status_placeholder.info("No hand detected.")

        if letter_detected:
            process_prediction_continuous(letter_detected, confidence_detected)
        else:
            check_auto_word_break()

        frame_window.image(frame_rgb)
        time.sleep(0.01)

    cap.release()


# ----------------------------------------------------------------------
# STREAMLIT UI
# ----------------------------------------------------------------------
def main():
    st.set_page_config(page_title="ISL Sign Language Translator", layout="wide")
    st.title("🤟 Real-Time ISL Sign Language Translator")

    init_state()

    model, classes, tf, device = load_model()
    hands = load_hand_detector()

    if model is None:
        st.warning(
            f"⚠️ Model file '{MODEL_PATH}' not found in this directory. "
            f"The app is running with placeholder predictions until you add it."
        )

    st.sidebar.header("⚙️ Settings")

    prediction_mode = st.sidebar.radio(
        "Prediction Mode",
        options=["Capture Mode (recommended)", "Continuous Mode"],
        index=0,
        help=(
            "Capture Mode: take one photo, get a prediction with confidence — "
            "much easier to debug misclassifications. "
            "Continuous Mode: live frame-by-frame prediction (original behavior)."
        ),
    )

    st.sidebar.checkbox(
        "Remove background before predicting",
        value=True,
        key="use_bg_removal",
        help=(
            "Strips walls, faces, and clutter from the captured hand crop and "
            "composites it onto a plain white background, matching the training "
            "data's style. Turn off to compare raw-crop predictions directly."
        ),
    )

    st.sidebar.checkbox(
        "Smart fallback crop (for A, K, Q, S, X, 4…)",
        value=True,
        key="use_fallback_crop",
        help=(
            "When MediaPipe cannot detect a hand (common for signs that use two hands "
            "like A, X, or tight fists like S, K, Q), fall back to a centred square "
            "crop of the camera frame and still run the model. Disable this if you "
            "want strict hand-detection-only predictions."
        ),
    )

    cam_index = st.sidebar.selectbox(
        "Camera Input Device (Continuous Mode only)",
        options=[0, 1, 2],
        index=0,
        help="0 is usually your built-in webcam. If your phone's camera keeps taking over, try 1 or 2, or disable Continuity Camera on iPhone.",
    )

    tts_mode = st.sidebar.selectbox(
        "Audio System (TTS Engine)",
        options=["Mac Native (say)", "Pyttsx3 Engine"],
        index=0,
    )

    col_video, col_text = st.columns([2, 1])

    with col_video:
        if prediction_mode == "Capture Mode (recommended)":
            run_capture_mode(model, tf, device, classes, hands)
        else:
            run = st.checkbox("Start Camera", value=st.session_state.camera_on)
            st.session_state.camera_on = run
            if st.session_state.camera_on:
                run_continuous_mode(model, tf, device, classes, hands, cam_index)

    with col_text:
        st.subheader("📝 Live Translation Output")

        st.markdown("**Current Spelled Word:**")
        st.info(st.session_state.current_word if st.session_state.current_word else "(Waiting for signs...)")

        st.markdown("**Constructed Sentence:**")
        st.success(st.session_state.sentence if st.session_state.sentence else "(Empty)")

        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Space / Commit Word"):
                finalize_word()
                st.rerun()
        with c2:
            if st.button("🔊 Speak Sentence"):
                speak(st.session_state.sentence, mode=tts_mode)
        with c3:
            if st.button("🗑️ Clear All", on_click=clear_all):
                st.rerun()


if __name__ == "__main__":
    main()