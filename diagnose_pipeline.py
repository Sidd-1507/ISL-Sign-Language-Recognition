"""
diagnose_pipeline.py
====================
Three-experiment diagnostic to find EXACTLY where the train→webcam mismatch is.

Run:
    cd ISL_Sign_Language_Project
    python diagnose_pipeline.py

What this script does
---------------------
  Experiment 1 — Feed a RAW test dataset image through the model.
                  If this fails → model itself is broken.
                  If this passes → model is fine, preprocessing is the bug.

  Experiment 2 — Feed the same image AFTER MediaPipe hand-crop (exactly the
                  inference path used by sign_language_app.py).
                  Compares the prediction & confidence with Experiment 1.

  Experiment 3 — Side-by-side visual comparison: original image, raw resize,
                  MediaPipe crop.  Saved as  debug_comparison.png.

All output goes to the terminal AND to  diagnosis_report.txt
"""

import os, sys, textwrap, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms, models

# ─────────────────────────────────────────────────────────────────────────────
# PATHS  (edit if needed)
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH      = "signformer_final.pth"
SPLIT_TRAIN_DIR = "dataset/split_dataset/train"   # dataset the model was trained on
HAND_TRAIN_DIR  = "dataset/hand_dataset/train"    # MediaPipe-cropped dataset
REPORT_FILE     = "diagnosis_report.txt"
DEBUG_IMG       = "debug_comparison.png"

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
lines = []

def log(msg=""):
    print(msg)
    lines.append(msg)

def save_report():
    with open(REPORT_FILE, "w") as f:
        f.write("\n".join(lines))
    print(f"\n[Report saved → {REPORT_FILE}]")

# ─────────────────────────────────────────────────────────────────────────────
# LOAD MODEL
# ─────────────────────────────────────────────────────────────────────────────
def load_model():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt   = torch.load(MODEL_PATH, map_location=device)
    classes = ckpt["class_names"]
    img_size = ckpt["image_size"]
    mean     = ckpt["mean"]
    std      = ckpt["std"]

    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device).eval()

    tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    return model, classes, tf, device, img_size, mean, std

# ─────────────────────────────────────────────────────────────────────────────
# PREDICT
# ─────────────────────────────────────────────────────────────────────────────
def predict(model, tf, device, classes, pil_img, k=3):
    """Returns list[(label, confidence)] top-k."""
    x = tf(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    top_probs, top_idxs = probs.topk(k)
    return [(classes[i.item()], p.item()) for p, i in zip(top_probs, top_idxs)]

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE CROP  (same code as sign_language_app.py)
# ─────────────────────────────────────────────────────────────────────────────
def mediapipe_crop(frame_rgb, margin_ratio=0.15):
    """
    Returns (crop_rgb, was_clipped, detected).
    crop_rgb is None if no hand detected.
    """
    import mediapipe as mp
    mp_hands = mp.solutions.hands
    with mp_hands.Hands(static_image_mode=True, max_num_hands=1,
                         min_detection_confidence=0.4) as hands:
        results = hands.process(frame_rgb)

    if not results.multi_hand_landmarks:
        return None, False, False

    lm    = results.multi_hand_landmarks[0]
    h, w  = frame_rgb.shape[:2]
    xs    = [p.x * w for p in lm.landmark]
    ys    = [p.y * h for p in lm.landmark]
    hw    = max(xs) - min(xs)
    hh    = max(ys) - min(ys)
    mx    = hw * margin_ratio
    my    = hh * margin_ratio

    x0r, x1r = min(xs) - mx, max(xs) + mx
    y0r, y1r = min(ys) - my, max(ys) + my
    x0 = max(0, int(x0r)); x1 = min(w, int(x1r))
    y0 = max(0, int(y0r)); y1 = min(h, int(y1r))

    if x1 <= x0 or y1 <= y0:
        return None, False, True

    clipped = (x0r < 0 or y0r < 0 or x1r > w or y1r > h)
    return frame_rgb[y0:y1, x0:x1], clipped, True

# ─────────────────────────────────────────────────────────────────────────────
# COLLECT SAMPLE IMAGES  (one per class, prefer split_dataset)
# ─────────────────────────────────────────────────────────────────────────────
def collect_samples(base_dir, max_per_class=1):
    samples = {}   # label → [path, ...]
    for cls in sorted(os.listdir(base_dir)):
        cls_dir = os.path.join(base_dir, cls)
        if not os.path.isdir(cls_dir): continue
        imgs = [os.path.join(cls_dir, f)
                for f in sorted(os.listdir(cls_dir))
                if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if imgs:
            samples[cls] = imgs[:max_per_class]
    return samples

# ─────────────────────────────────────────────────────────────────────────────
# MAKE COMPARISON IMAGE
# ─────────────────────────────────────────────────────────────────────────────
def make_comparison_png(orig_pil, raw_resize_pil, mp_crop_pil, label,
                         exp1_preds, exp2_preds, out_path):
    """3-panel side-by-side with predictions labelled."""
    W, H = 224, 224
    gap  = 20
    header = 50
    total_w = 3 * W + 4 * gap
    total_h = H + header + gap * 2

    canvas = Image.new("RGB", (total_w, total_h), (30, 30, 30))
    draw   = ImageDraw.Draw(canvas)

    def paste(img, x, title):
        img_r = img.resize((W, H))
        canvas.paste(img_r, (x, header))
        draw.text((x, 5), title, fill=(220, 220, 220))

    paste(orig_pil,      gap,             f"[1] Original ({label})")
    paste(raw_resize_pil, gap + W + gap,  f"[2] Raw resize 224")
    if mp_crop_pil is not None:
        paste(mp_crop_pil,  gap + 2*(W+gap), f"[3] MediaPipe crop")
    else:
        draw.text((gap + 2*(W+gap), header + H//2),
                  "No hand\ndetected", fill=(255, 80, 80))

    # Write predictions below
    def fmt(preds):
        return " | ".join(f"{c}:{p*100:.1f}%" for c, p in preds)

    y = header + H + 5
    draw.text((gap, y), f"Exp1: {fmt(exp1_preds)}", fill=(100, 255, 100))
    draw.text((gap, y + 18), f"Exp2: {fmt(exp2_preds) if exp2_preds else 'N/A'}", fill=(255, 200, 80))

    canvas.save(out_path)

# ─────────────────────────────────────────────────────────────────────────────
# MAIN DIAGNOSTIC
# ─────────────────────────────────────────────────────────────────────────────
def main():
    log("=" * 70)
    log("  ISL PIPELINE DIAGNOSTIC — Train vs Webcam Mismatch Finder")
    log("=" * 70)

    # ── Load model ──────────────────────────────────────────────────────────
    log("\n[Loading model …]")
    model, classes, tf, device, img_size, mean, std = load_model()
    log(f"  Device       : {device}")
    log(f"  Classes ({len(classes)}): {classes}")
    log(f"  img_size     : {img_size}")
    log(f"  mean / std   : {mean} / {std}")

    # ── Collect samples ──────────────────────────────────────────────────────
    log("\n[Collecting dataset samples …]")
    split_samples = collect_samples(SPLIT_TRAIN_DIR)
    hand_samples  = collect_samples(HAND_TRAIN_DIR)

    split_classes = set(split_samples.keys())
    hand_classes  = set(hand_samples.keys())
    common_cls    = sorted(split_classes & hand_classes)

    log(f"  split_dataset classes : {sorted(split_classes)}")
    log(f"  hand_dataset classes  : {sorted(hand_classes)}")
    log(f"  model class_names     : {classes}")
    log(f"  Classes IN split but NOT in hand : {sorted(split_classes - hand_classes)}")
    log(f"  Classes IN hand  but NOT in split: {sorted(hand_classes - split_classes)}")

    # ── Check class names match model ────────────────────────────────────────
    model_cls_set = set(classes)
    if split_classes.issubset(model_cls_set):
        log("\n  ✓ split_dataset classes are a SUBSET of model classes — consistent.")
    else:
        log(f"\n  ✗ MISMATCH: split classes {sorted(split_classes - model_cls_set)} "
            "not in model's class list!")

    # ── EXPERIMENT 1 & 2  (per class) ──────────────────────────────────────
    log("\n" + "=" * 70)
    log("  EXPERIMENT 1  — Dataset image → model (no MediaPipe)")
    log("  EXPERIMENT 2  — Dataset image → MediaPipe crop → model")
    log("=" * 70)

    results = []   # (label, exp1_top1, exp1_conf, exp2_top1, exp2_conf, mp_detected, mp_clipped)

    # Use split_dataset as ground truth; if a class isn't in split_dataset use hand_dataset
    all_cls_to_test = sorted(split_classes | hand_classes)

    for cls in all_cls_to_test:
        img_path = None
        source   = None
        if cls in split_samples:
            img_path = split_samples[cls][0]
            source   = "split_dataset"
        elif cls in hand_samples:
            img_path = hand_samples[cls][0]
            source   = "hand_dataset"

        orig_pil  = Image.open(img_path).convert("RGB")
        frame_rgb = np.array(orig_pil)

        # --- Exp 1: raw resize ---
        exp1_preds = predict(model, tf, device, classes, orig_pil)
        exp1_top1, exp1_conf = exp1_preds[0]
        exp1_correct = (exp1_top1 == cls)

        # --- Exp 2: MediaPipe crop ---
        crop_rgb, clipped, detected = mediapipe_crop(frame_rgb)
        exp2_preds = None
        exp2_top1  = "N/A"
        exp2_conf  = 0.0
        exp2_correct = False
        if detected and crop_rgb is not None:
            crop_pil   = Image.fromarray(crop_rgb)
            exp2_preds = predict(model, tf, device, classes, crop_pil)
            exp2_top1, exp2_conf = exp2_preds[0]
            exp2_correct = (exp2_top1 == cls)

        label_exp1 = "✓" if exp1_correct else "✗"
        label_exp2 = ("✓" if exp2_correct else "✗") if detected else "⚠ no hand"

        log(f"\n  Class={cls:<3}  source={source}")
        log(f"    Exp1 (raw)  → {exp1_top1} ({exp1_conf*100:.1f}%)  {label_exp1}")
        if detected:
            log(f"    Exp2 (crop) → {exp2_top1} ({exp2_conf*100:.1f}%)  {label_exp2}"
                + ("  [CLIPPED]" if clipped else ""))
        else:
            log(f"    Exp2 (crop) → MediaPipe found NO hand ⚠")

        results.append((cls, exp1_top1, exp1_conf, exp1_correct,
                         exp2_top1, exp2_conf, exp2_correct, detected, clipped, source))

        # Save a comparison PNG for the FIRST class only (you can extend)
        if cls == all_cls_to_test[0]:
            raw_resize_pil = orig_pil.resize((224, 224))
            mp_crop_pil    = Image.fromarray(crop_rgb) if (detected and crop_rgb is not None) else None
            make_comparison_png(
                orig_pil, raw_resize_pil, mp_crop_pil, cls,
                exp1_preds, exp2_preds or [],
                DEBUG_IMG
            )
            log(f"    [Comparison image saved → {DEBUG_IMG}]")

    # ── SUMMARY ─────────────────────────────────────────────────────────────
    log("\n" + "=" * 70)
    log("  SUMMARY")
    log("=" * 70)

    exp1_correct_count  = sum(1 for r in results if r[3])
    exp2_detected       = [r for r in results if r[7]]
    exp2_correct_count  = sum(1 for r in exp2_detected if r[6])
    mp_no_hand          = [r[0] for r in results if not r[7]]
    mp_clipped          = [r[0] for r in results if r[8]]

    total = len(results)
    log(f"\n  Total classes tested          : {total}")
    log(f"  Exp1 correct (raw resize)     : {exp1_correct_count}/{total} "
        f"= {exp1_correct_count/total*100:.1f}%")
    log(f"  MediaPipe detected hand in    : {len(exp2_detected)}/{total}")
    if exp2_detected:
        log(f"  Exp2 correct (crop→model)     : {exp2_correct_count}/{len(exp2_detected)} "
            f"= {exp2_correct_count/len(exp2_detected)*100:.1f}%")
    if mp_no_hand:
        log(f"  ⚠ No hand detected in classes : {mp_no_hand}")
    if mp_clipped:
        log(f"  ⚠ Hand clipped at frame edge  : {mp_clipped}")

    # ── DIAGNOSIS ───────────────────────────────────────────────────────────
    log("\n" + "─" * 70)
    log("  DIAGNOSIS")
    log("─" * 70)

    if exp1_correct_count < total * 0.7:
        log("""
  ✗ LOW Exp1 accuracy — The model is not working well even on raw test
    images.  Possible causes:
    • split_dataset has classes the model was NOT trained on
    • The model was trained on hand_dataset (MediaPipe crops) but you are
      loading signformer_final.pth which may be the split_dataset version
    → ACTION: check which dataset was used to produce signformer_final.pth
""")
    elif exp2_correct_count < exp1_correct_count * 0.8:
        drop = exp1_correct_count - exp2_correct_count
        log(f"""
  ✗ CONFIRMED MISMATCH: Accuracy drops {drop} class(es) after MediaPipe crop.
    The model works fine on raw images but fails on cropped images.
    Root cause: TRAIN–INFERENCE DISTRIBUTION MISMATCH.

    During training:
        Full image (300×300) → Resize 224 → Normalize → ViT

    During webcam inference:
        Full image → MediaPipe → Small crop (varies!) → Resize 224 → ViT

    The hand appears at a DIFFERENT scale/context than during training.
    → ACTION: Re-train on MediaPipe-cropped images (hand_dataset).
              Use the pipeline_fix_train.py script.
""")
    elif mp_no_hand:
        log(f"""
  ⚠ MediaPipe CANNOT detect hands in some dataset images.
    This means your training images may not look like webcam frames.
    Classes affected: {mp_no_hand}
    → ACTION: These classes need real webcam data collected and
              re-cropped with MediaPipe for fine-tuning.
""")
    else:
        log("""
  ✓ Both experiments work well.  If webcam still fails, the issue is
    live-webcam specific:
    • Lighting conditions
    • Camera resolution / blur
    • Hand too far / too close
    → ACTION: Try the confidence_debug.py to log live webcam predictions.
""")

    # ── WHAT TO DO NEXT ─────────────────────────────────────────────────────
    log("\n" + "─" * 70)
    log("  NEXT STEPS (in order)")
    log("─" * 70)
    log("""
  1. Open  debug_comparison.png  and visually inspect:
       Panel 1 = original dataset image
       Panel 2 = naïve 224 resize (what model saw during training)
       Panel 3 = MediaPipe crop (what model sees at inference)
     If they look dramatically different → mismatch confirmed.

  2. If mismatch confirmed → run  pipeline_fix_train.py  to fine-tune
     the existing model on MediaPipe-cropped images (hand_dataset).
     This is fast (only fine-tuning the head, ~10–15 min).

  3. After fine-tuning, replace  signformer_final.pth  with the new
     checkpoint.  sign_language_app.py will automatically pick it up.

  4. Only if accuracy is still poor after (2) → generate more data
     using the synthetic data pipeline.
""")

    save_report()

if __name__ == "__main__":
    main()
