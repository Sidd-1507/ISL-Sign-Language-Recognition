"""
pipeline_fix_train.py
=====================
Fixes the train-inference mismatch by:

  Step 1 — Re-crop every training image with MediaPipe (same as webcam does).
            For images where MediaPipe finds no hand, use smart_center_crop.
            Saves crops to  dataset/webcam_ready_dataset/

  Step 2 — Fine-tune signformer_final.pth on the re-cropped images.
            Only the classification head is unfrozen for the first half
            of training (fast + stable), then the full network is unfrozen
            for the second half (for best accuracy).

  Step 3 — Save the new model as  signformer_webcam_finetuned.pth
            (sign_language_app.py auto-reads it if you rename it to
             signformer_final.pth)

Run:
    source venv311/bin/activate
    python pipeline_fix_train.py
"""

import os, json, shutil, time, warnings
warnings.filterwarnings("ignore")

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms, models
from torch.utils.data import DataLoader, Dataset, random_split

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────────────────────────────────────
MODEL_IN        = "signformer_final.pth"
MODEL_OUT       = "signformer_webcam_finetuned.pth"
WEBCAM_DS_DIR   = "dataset/webcam_ready_dataset"

# Source datasets — we merge both
SOURCE_DIRS = [
    "dataset/split_dataset",   # original 300×300 images
    "dataset/hand_dataset",    # already MediaPipe-cropped images
]

IMG_SIZE    = 224
BATCH_SIZE  = 32
EPOCHS_HEAD = 8    # train only the head first (fast, stable)
EPOCHS_FULL = 12   # then unfreeze all layers (deeper adjustment)
LR_HEAD     = 3e-4
LR_FULL     = 5e-5
WEIGHT_DECAY= 1e-4
VAL_FRAC    = 0.15  # fraction of re-cropped data used for validation

# ─────────────────────────────────────────────────────────────────────────────
# MEDIAPIPE CROP + SMART FALLBACK
# ─────────────────────────────────────────────────────────────────────────────
def mediapipe_crop(frame_rgb, hands, margin_ratio=0.20):
    """Returns (crop_rgb, method) where method is 'mediapipe' or 'fallback'."""
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        # Use the first detected hand
        lm  = results.multi_hand_landmarks[0]
        h, w = frame_rgb.shape[:2]
        xs  = [p.x * w for p in lm.landmark]
        ys  = [p.y * h for p in lm.landmark]
        hw  = max(xs) - min(xs)
        hh  = max(ys) - min(ys)
        mx  = hw * margin_ratio
        my  = hh * margin_ratio

        x0  = max(0, int(min(xs) - mx))
        x1  = min(w, int(max(xs) + mx))
        y0  = max(0, int(min(ys) - my))
        y1  = min(h, int(max(ys) + my))

        if x1 > x0 and y1 > y0:
            return frame_rgb[y0:y1, x0:x1], "mediapipe"

    # Smart center-crop fallback
    h, w = frame_rgb.shape[:2]
    side  = int(min(h, w) * 0.70)
    cx, cy = w // 2, h // 2
    x0 = max(0, cx - side // 2); x1 = min(w, x0 + side)
    y0 = max(0, cy - side // 2); y1 = min(h, y0 + side)
    return frame_rgb[y0:y1, x0:x1], "fallback"


# ─────────────────────────────────────────────────────────────────────────────
# STEP 1 — RE-CROP ALL IMAGES
# ─────────────────────────────────────────────────────────────────────────────
def recrop_all_images(source_dirs, out_dir):
    if os.path.exists(out_dir):
        print(f"  {out_dir} already exists — skipping re-crop (delete to redo).")
        return

    print(f"\n{'='*60}")
    print("  STEP 1 — Re-cropping dataset images with MediaPipe")
    print(f"{'='*60}")

    total, mp_count, fb_count, skip_count = 0, 0, 0, 0

    import mediapipe as mp
    mp_hands = mp.solutions.hands
    
    # Use integer 1 for LANCZOS to avoid Pillow 10+ AttributeError
    LANCZOS_RESAMPLE = 1 
    if hasattr(Image, "Resampling"):
        LANCZOS_RESAMPLE = Image.Resampling.LANCZOS
    elif hasattr(Image, "LANCZOS"):
        LANCZOS_RESAMPLE = Image.LANCZOS

    with mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.35) as hands:
        for src_root in source_dirs:
            for split in ["train", "val", "test"]:
                split_dir = os.path.join(src_root, split)
                if not os.path.isdir(split_dir):
                    continue
                for cls in sorted(os.listdir(split_dir)):
                    cls_dir = os.path.join(split_dir, cls)
                    if not os.path.isdir(cls_dir):
                        continue

                    out_cls = os.path.join(out_dir, split, cls)
                    os.makedirs(out_cls, exist_ok=True)

                    imgs = [f for f in sorted(os.listdir(cls_dir))
                            if f.lower().endswith((".jpg", ".jpeg", ".png"))]

                    for fname in imgs:
                        src_path = os.path.join(cls_dir, fname)
                        # avoid filename collision when merging two source dirs
                        ds_prefix = "sp" if "split_dataset" in src_root else "hd"
                        out_name  = f"{ds_prefix}_{cls}_{fname}"
                        out_path  = os.path.join(out_cls, out_name)

                        if os.path.exists(out_path):
                            continue

                        try:
                            img_pil   = Image.open(src_path).convert("RGB")
                            frame_rgb = np.array(img_pil)
                            crop, method = mediapipe_crop(frame_rgb, hands)
                            crop_pil  = Image.fromarray(crop).resize((IMG_SIZE, IMG_SIZE), LANCZOS_RESAMPLE)
                            crop_pil.save(out_path, quality=95)
                            if method == "mediapipe":
                                mp_count += 1
                            else:
                                fb_count += 1
                            total += 1
                            if total % 1000 == 0:
                                print(f"    ... processed {total} images so far ...")
                        except Exception as e:
                            skip_count += 1
                            continue

    print(f"\n  Done!")
    print(f"  Total saved  : {total}")
    print(f"  MediaPipe    : {mp_count}")
    print(f"  Fallback crop: {fb_count}")
    print(f"  Skipped      : {skip_count}")


# ─────────────────────────────────────────────────────────────────────────────
# DATASET CLASS
# ─────────────────────────────────────────────────────────────────────────────
class CroppedHandDataset(Dataset):
    def __init__(self, root, class_names, transform):
        self.transform    = transform
        self.class_names  = class_names
        self.cls_to_idx   = {c: i for i, c in enumerate(class_names)}
        self.samples      = []

        for cls in sorted(os.listdir(root)):
            cls_dir = os.path.join(root, cls)
            if not os.path.isdir(cls_dir) or cls not in self.cls_to_idx:
                continue
            for f in sorted(os.listdir(cls_dir)):
                if f.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.samples.append((os.path.join(cls_dir, f),
                                         self.cls_to_idx[cls]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING UTILS
# ─────────────────────────────────────────────────────────────────────────────
def accuracy(outputs, labels):
    preds = outputs.argmax(dim=1)
    return (preds == labels).float().mean().item()


def run_epoch(model, loader, criterion, optimizer, device, train=True):
    model.train(train)
    total_loss, total_acc, n = 0.0, 0.0, 0
    with torch.set_grad_enabled(train):
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            if train:
                optimizer.zero_grad()
            out  = model(imgs)
            loss = criterion(out, labels)
            if train:
                loss.backward()
                optimizer.step()
            bs = imgs.size(0)
            total_loss += loss.item() * bs
            total_acc  += accuracy(out, labels) * bs
            n          += bs
    return total_loss / n, total_acc / n


# ─────────────────────────────────────────────────────────────────────────────
# STEP 2 — FINE-TUNE MODEL
# ─────────────────────────────────────────────────────────────────────────────
def finetune(webcam_ds_dir, model_in, model_out):
    print(f"\n{'='*60}")
    print("  STEP 2 — Fine-tuning model on webcam-cropped dataset")
    print(f"{'='*60}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device: {device}")

    # Load original checkpoint
    ckpt    = torch.load(model_in, map_location=device)
    classes = ckpt["class_names"]
    mean    = ckpt["mean"]
    std     = ckpt["std"]
    print(f"  Loaded {model_in}  ({len(classes)} classes)")

    # Transforms — SAME normalization as original, but now with augmentation
    # during training that is SAFE (no flips/rotations that change meaning).
    train_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE + 16, IMG_SIZE + 16)),
        transforms.RandomCrop(IMG_SIZE),              # slight translation
        transforms.ColorJitter(brightness=0.3,
                                contrast=0.3,
                                saturation=0.2),      # lighting variation
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])
    val_tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])

    # Build datasets
    train_dir = os.path.join(webcam_ds_dir, "train")
    # Collect images from both train and val source dirs for fine-tuning
    # (we re-split since the original val set is small)
    all_sources = []
    for split in ["train", "val"]:
        sdir = os.path.join(webcam_ds_dir, split)
        if os.path.isdir(sdir):
            all_sources.append(sdir)

    # Simple approach: use train/ dir from webcam_ready_dataset
    full_ds = CroppedHandDataset(train_dir, classes, train_tf)
    val_len  = max(1, int(len(full_ds) * VAL_FRAC))
    trn_len  = len(full_ds) - val_len
    trn_ds, val_ds = random_split(full_ds, [trn_len, val_len],
                                   generator=torch.Generator().manual_seed(42))
    # val uses val_tf — monkey-patch transform
    val_ds.dataset = CroppedHandDataset(train_dir, classes, val_tf)

    print(f"  Train samples: {len(trn_ds)} | Val samples: {len(val_ds)}")

    trn_loader = DataLoader(trn_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=0, pin_memory=False)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=0)

    # Build model
    model = models.vit_b_16(weights=None)
    model.heads.head = nn.Linear(model.heads.head.in_features, len(classes))
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    best_val_acc = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}

    # ── Phase 1: Train only the head ──────────────────────────────────────
    print(f"\n  Phase 1 — Head-only ({EPOCHS_HEAD} epochs, LR={LR_HEAD})")
    for p in model.parameters():
        p.requires_grad = False
    for p in model.heads.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=LR_HEAD, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_HEAD)

    for ep in range(EPOCHS_HEAD):
        t0 = time.time()
        tl, ta = run_epoch(model, trn_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"    Ep {ep+1:02}/{EPOCHS_HEAD}  train={ta*100:.1f}%  val={va*100:.1f}%  "
              f"({elapsed:.0f}s)")
        history["train_loss"].append(tl); history["train_acc"].append(ta * 100)
        history["val_loss"].append(vl);   history["val_acc"].append(va * 100)
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), "_best_state_dict.pth")

    # ── Phase 2: Fine-tune all layers ─────────────────────────────────────
    print(f"\n  Phase 2 — Full network ({EPOCHS_FULL} epochs, LR={LR_FULL})")
    for p in model.parameters():
        p.requires_grad = True

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR_FULL,
                                   weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, EPOCHS_FULL)

    for ep in range(EPOCHS_FULL):
        t0 = time.time()
        tl, ta = run_epoch(model, trn_loader, criterion, optimizer, device, train=True)
        vl, va = run_epoch(model, val_loader, criterion, optimizer, device, train=False)
        scheduler.step()
        elapsed = time.time() - t0
        print(f"    Ep {ep+1:02}/{EPOCHS_FULL}  train={ta*100:.1f}%  val={va*100:.1f}%  "
              f"({elapsed:.0f}s)")
        history["train_loss"].append(tl); history["train_acc"].append(ta * 100)
        history["val_loss"].append(vl);   history["val_acc"].append(va * 100)
        if va > best_val_acc:
            best_val_acc = va
            torch.save(model.state_dict(), "_best_state_dict.pth")

    # ── Save final checkpoint ─────────────────────────────────────────────
    model.load_state_dict(torch.load("_best_state_dict.pth", map_location=device))
    os.remove("_best_state_dict.pth")

    new_ckpt = {
        "model_state_dict": model.state_dict(),
        "class_names":      classes,
        "image_size":       IMG_SIZE,
        "mean":             mean,
        "std":              std,
    }
    torch.save(new_ckpt, model_out)
    print(f"\n  ✓ Best val acc: {best_val_acc*100:.2f}%")
    print(f"  ✓ Model saved → {model_out}")

    with open("finetune_history.json", "w") as f:
        json.dump(history, f, indent=2)
    print(f"  ✓ History  → finetune_history.json")

    print(f"""
  ┌─────────────────────────────────────────────────────┐
  │  NEXT STEP                                          │
  │                                                     │
  │  Replace your current model with the new one:       │
  │    mv signformer_final.pth signformer_original.pth  │
  │    cp {model_out} signformer_final.pth              │
  │                                                     │
  │  Then restart the Streamlit app:                    │
  │    streamlit run sign_language_app.py               │
  └─────────────────────────────────────────────────────┘
""")


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    recrop_all_images(SOURCE_DIRS, WEBCAM_DS_DIR)
    finetune(WEBCAM_DS_DIR, MODEL_IN, MODEL_OUT)
