# -*- coding: utf-8 -*-
"""
Tea Leaf Disease Classification — v5.1

Two-phase training pipeline for EfficientNet-B3:
  Phase 1: Head training at 224×224 (frozen backbone)
  Phase 2: Fine-tuning at 300×300 (top 60 layers unfrozen)

Techniques: Focal Loss, MixUp, background-swap regularisation,
cosine LR decay, class-weighted training.

Designed to run on Google Colab with GPU.
"""


import os
import sys
import gc
import cv2
import json
import random
import datetime
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats
from pathlib import Path
from PIL import Image
from tqdm.auto import tqdm

import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                              f1_score, roc_auc_score)
from sklearn.utils.class_weight import compute_class_weight

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'

print(f"Python     : {sys.version.split()[0]}")
print(f"TensorFlow : {tf.__version__}")
print(f"OpenCV     : {cv2.__version__}")
print(f"GPU        : {tf.config.list_physical_devices('GPU')}")

from google.colab import drive
drive.mount('/content/drive')


MASTER_DIR = "/content/drive/MyDrive/TeaDataset/master"
BASE_DIR   = "/content/drive/MyDrive/TeaDataset"

PREV_CROP_RUN = None  # Example: "/content/drive/MyDrive/TeaDataset/v5.0/run_20260302_105820"

timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
RUN_DIR   = os.path.join(BASE_DIR, "v5.1", f"run_{timestamp}")
os.makedirs(os.path.join(RUN_DIR, "swapped_crops"), exist_ok=True)
print(f"\nRun dir: {RUN_DIR}")

SELECTED_CLASSES = ['Algal Leaf', 'Healthy', 'Red Leaf Spot', 'White Spot']
classes          = sorted(SELECTED_CLASSES)
class_indices    = {cls: i for i, cls in enumerate(classes)}
num_classes      = len(classes)

EXPAND_FACTOR  = 1.4
IMG_SIZE_S1    = (224, 224)
IMG_SIZE_S2    = (300, 300)
BATCH_SIZE     = 16
EPOCHS_HEAD    = 10
EPOCHS_FINE    = 30
LR_HEAD        = 1e-3
LR_FINE        = 5e-5
WEIGHT_DECAY   = 1e-4
LABEL_SMOOTH   = 0.1
FOCAL_GAMMA    = 2.0
DROPOUT_RATE   = 0.4
UNFREEZE_TOP   = 60
EARLY_STOP_PAT = 8
MIXUP_ALPHA    = 0.2

with open(os.path.join(RUN_DIR, "class_indices.json"), 'w') as f:
    json.dump(class_indices, f, indent=2)

cfg = dict(SELECTED_CLASSES=SELECTED_CLASSES, EXPAND_FACTOR=EXPAND_FACTOR,
           IMG_SIZE_S1=list(IMG_SIZE_S1), IMG_SIZE_S2=list(IMG_SIZE_S2),
           BATCH_SIZE=BATCH_SIZE, EPOCHS_HEAD=EPOCHS_HEAD,
           EPOCHS_FINE=EPOCHS_FINE, LR_HEAD=LR_HEAD, LR_FINE=LR_FINE,
           SEED=SEED)
with open(os.path.join(RUN_DIR, "config.json"), 'w') as f:
    json.dump(cfg, f, indent=2)


print("\n--- EDA ---")
class_dirs = {cls: os.path.join(MASTER_DIR, cls) for cls in classes}
file_rows  = []
for cls in classes:
    imgs = [f for f in os.listdir(class_dirs[cls])
            if f.lower().endswith(('.jpg','.jpeg','.png'))]
    print(f"  {cls:20s}: {len(imgs)} images")
    for f in imgs:
        file_rows.append({'class': cls,
                          'path': os.path.join(class_dirs[cls], f)})
df_all = pd.DataFrame(file_rows)

# Sample grid
fig, axes = plt.subplots(num_classes, 4, figsize=(16, 4*num_classes))
for r, cls in enumerate(classes):
    sub = df_all[df_all['class']==cls].sample(
        min(4, len(df_all[df_all['class']==cls])), random_state=SEED)
    for c, (_, row) in enumerate(sub.iterrows()):
        img = np.array(Image.open(row['path']).convert('RGB').resize((200,200)))
        axes[r,c].imshow(img)
        axes[r,c].set_title(cls, fontsize=7)
        axes[r,c].axis('off')
plt.suptitle("EDA — Sample Images", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "eda_samples.png"), dpi=150)
plt.show()


def extract_leaf_bbox_and_crop(img_bgr, expand=EXPAND_FACTOR, min_area=500):
    """
    HSV leaf localization. H upper bound = 90 (not 100) to exclude
    blue backgrounds. Otsu fallback when >85% of image is flagged.
    """
    hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    mask  = cv2.inRange(hsv, np.array([25, 25, 25]), np.array([90, 255, 255]))
    mask2 = cv2.inRange(hsv, np.array([10, 30, 30]), np.array([30, 255, 255]))
    mask  = cv2.bitwise_or(mask, mask2)
    k     = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask  = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=2)
    mask  = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  k, iterations=1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, img_bgr.copy(), mask

    c = max(contours, key=cv2.contourArea)
    if cv2.contourArea(c) < min_area:
        return None, img_bgr.copy(), mask

    ratio = float((mask > 0).sum()) / mask.size
    if ratio > 0.85:
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 0, 255,
                                cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        k2 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k2, iterations=3)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None, img_bgr.copy(), mask
        c = max(contours, key=cv2.contourArea)

    x, y, w, h = cv2.boundingRect(c)
    cx, cy = x + w / 2, y + h / 2
    x1 = int(max(0, cx - w * expand / 2))
    y1 = int(max(0, cy - h * expand / 2))
    x2 = int(min(img_bgr.shape[1], cx + w * expand / 2))
    y2 = int(min(img_bgr.shape[0], cy + h * expand / 2))
    return (x1, y1, x2, y2), img_bgr[y1:y2, x1:x2].copy(), mask


def neutralize_background(img_bgr, mask):
    result = img_bgr.copy()
    result[mask == 0] = 128
    return result


# Localization preview
print("\nLocalization preview (check gray bg + tight crop)...")
fig, axes = plt.subplots(num_classes, 4, figsize=(20, 5 * num_classes))
for r, cls in enumerate(classes):
    row     = df_all[df_all['class'] == cls].sample(1, random_state=SEED).iloc[0]
    img_bgr = cv2.resize(cv2.imread(row['path']), (600, 600))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    bbox, crop, mask = extract_leaf_bbox_and_crop(img_bgr)
    crop_mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                           interpolation=cv2.INTER_NEAREST)
    neutral = neutralize_background(img_bgr, mask)
    crop_n  = neutralize_background(crop, crop_mask)
    axes[r, 0].imshow(img_rgb);                                       axes[r,0].set_title(f"{cls}\nOriginal");    axes[r,0].axis('off')
    axes[r, 1].imshow(mask, cmap='gray');                             axes[r,1].set_title("HSV mask");            axes[r,1].axis('off')
    axes[r, 2].imshow(cv2.cvtColor(neutral, cv2.COLOR_BGR2RGB));      axes[r,2].set_title("BG neutralized");      axes[r,2].axis('off')
    axes[r, 3].imshow(cv2.cvtColor(crop_n, cv2.COLOR_BGR2RGB));       axes[r,3].set_title(f"Crop {EXPAND_FACTOR}x"); axes[r,3].axis('off')
plt.suptitle("Localization Preview — verify before proceeding", fontsize=12)
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "localization_preview.png"), dpi=150)
plt.show()


if (PREV_CROP_RUN and
        os.path.exists(os.path.join(PREV_CROP_RUN, "crop_manifest.csv"))):
    print(f"\nLoading crops from: {PREV_CROP_RUN}")
    crop_df = pd.read_csv(os.path.join(PREV_CROP_RUN, "crop_manifest.csv"))
    crop_df = crop_df[crop_df['scale'] == EXPAND_FACTOR].copy()
    print(f"  {len(crop_df)} crops loaded (1.4x only)")
else:
    print("\nGenerating crops from master dataset...")
    crop_rows = []
    for cls in classes:
        out_dir = os.path.join(RUN_DIR, "crops", cls)
        os.makedirs(out_dir, exist_ok=True)
        imgs = sorted([f for f in os.listdir(class_dirs[cls])
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
        for fname in tqdm(imgs, desc=f"  {cls}"):
            src     = os.path.join(class_dirs[cls], fname)
            img_bgr = cv2.imread(src)
            if img_bgr is None:
                continue
            img_bgr   = cv2.resize(img_bgr, (600, 600))
            bbox, crop, mask = extract_leaf_bbox_and_crop(img_bgr)
            crop_mask = cv2.resize(mask, (crop.shape[1], crop.shape[0]),
                                   interpolation=cv2.INTER_NEAREST)
            crop      = neutralize_background(crop, crop_mask)
            out_path  = os.path.join(out_dir, Path(fname).stem + ".jpg")
            cv2.imwrite(out_path, crop)
            crop_rows.append({'path': out_path, 'class': cls,
                              'scale': EXPAND_FACTOR, 'orig_path': src})
    crop_df = pd.DataFrame(crop_rows)
    crop_df.to_csv(os.path.join(RUN_DIR, "crop_manifest.csv"), index=False)

print("\nCrops by class:")
print(crop_df['class'].value_counts().sort_index().to_string())

# Splits
train_base, temp_df = train_test_split(
    crop_df, test_size=0.30, stratify=crop_df['class'], random_state=SEED)
val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df['class'], random_state=SEED)

train_base = train_base.reset_index(drop=True)
val_df     = val_df.reset_index(drop=True)
test_df    = test_df.reset_index(drop=True)
print(f"\nSplits — Train: {len(train_base)}  Val: {len(val_df)}  Test: {len(test_df)}")


# Build patch pool (8 per class = 32 patches, small by design)
print("\nBuilding background patch pool...")
bg_patches = []
for cls in classes:
    imgs = [f for f in os.listdir(class_dirs[cls])
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in random.sample(imgs, min(8, len(imgs))):
        img = cv2.imread(os.path.join(class_dirs[cls], fname))
        if img is None:
            continue
        img   = cv2.resize(img, (300, 300))
        hsv   = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fg    = cv2.inRange(hsv, np.array([25, 25, 25]),
                            np.array([90, 255, 255]))
        patch = img.copy()
        patch[fg > 0] = 128
        bg_patches.append(patch)
print(f"  {len(bg_patches)} patches ready")


def apply_random_bg_swap(img_bgr):
    donor   = bg_patches[random.randint(0, len(bg_patches) - 1)]
    donor_r = cv2.resize(donor, (img_bgr.shape[1], img_bgr.shape[0]))
    gray_px = np.all(img_bgr == 128, axis=-1)
    result  = img_bgr.copy()
    result[gray_px] = donor_r[gray_px]
    return result


# Write one swapped copy per training image to disk
SWAP_DIR  = os.path.join(RUN_DIR, "swapped_crops")
swap_rows = []
print("Pre-generating swapped crops on disk...")

for _, row in tqdm(train_base.iterrows(), total=len(train_base)):
    img_bgr = cv2.imread(row['path'])
    if img_bgr is None:
        continue
    swapped  = apply_random_bg_swap(img_bgr)
    out_dir  = os.path.join(SWAP_DIR, row['class'])
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, Path(row['path']).stem + "_swap.jpg")
    cv2.imwrite(out_path, swapped)
    swap_rows.append({'path': out_path, 'class': row['class'],
                      'scale': EXPAND_FACTOR,
                      'orig_path': row.get('orig_path', row['path'])})

swap_df  = pd.DataFrame(swap_rows)
train_df = pd.concat([train_base, swap_df], ignore_index=True)\
    .sample(frac=1, random_state=SEED).reset_index(drop=True)
print(f"  {len(train_base)} original + {len(swap_df)} swapped "
      f"= {len(train_df)} total training images")

# Verify
fig, ax = plt.subplots(1, 2, figsize=(8, 4))
ax[0].imshow(cv2.cvtColor(cv2.imread(train_base.iloc[0]['path']),
                           cv2.COLOR_BGR2RGB))
ax[0].set_title("Original (gray bg)"); ax[0].axis('off')
ax[1].imshow(cv2.cvtColor(cv2.imread(swap_df.iloc[0]['path']),
                           cv2.COLOR_BGR2RGB))
ax[1].set_title("Swapped bg"); ax[1].axis('off')
plt.suptitle("Background swap verification", fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "swap_verification.png"), dpi=150)
plt.show()

del bg_patches, swap_rows
gc.collect()


rotation_aug = layers.RandomRotation(factor=1.0, fill_mode='nearest')


def augment_train(image, label):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    image = rotation_aug(image)
    image = tf.cond(tf.random.uniform([]) < 0.2,
                    lambda: tf.image.random_brightness(image, 0.1),
                    lambda: image)
    image = tf.cond(tf.random.uniform([]) < 0.2,
                    lambda: tf.image.random_contrast(image, 0.9, 1.1),
                    lambda: image)
    return tf.clip_by_value(image, 0.0, 255.0), label


def mixup_batch(images, labels, alpha=MIXUP_ALPHA):
    batch_size = tf.shape(images)[0]
    g1  = -tf.math.log(tf.random.uniform([batch_size], 1e-8, 1.0))
    g2  = -tf.math.log(tf.random.uniform([batch_size], 1e-8, 1.0))
    lam = tf.clip_by_value(g1 / (g1 + g2), 0.2, 0.8)
    lam_img = tf.reshape(lam, [batch_size, 1, 1, 1])
    lam_lbl = tf.reshape(lam, [batch_size, 1])
    idx     = tf.random.shuffle(tf.range(batch_size))
    return (lam_img * images + (1 - lam_img) * tf.gather(images, idx),
            lam_lbl * labels + (1 - lam_lbl) * tf.gather(labels, idx))


class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, gamma=FOCAL_GAMMA, label_smoothing=LABEL_SMOOTH, **kw):
        super().__init__(**kw)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        n   = tf.cast(tf.shape(y_true)[-1], tf.float32)
        y_t = y_true * (1 - self.label_smoothing) + self.label_smoothing / n
        y_p = tf.clip_by_value(y_pred, 1e-7, 1 - 1e-7)
        ce  = -y_t * tf.math.log(y_p)
        pt  = tf.reduce_sum(y_t * y_p, axis=-1, keepdims=True)
        return tf.reduce_mean(
            tf.pow(1 - pt, self.gamma) * tf.reduce_sum(ce, axis=-1))

    def get_config(self):
        c = super().get_config()
        c.update({'gamma': self.gamma, 'label_smoothing': self.label_smoothing})
        return c

focal_loss = FocalLoss()


def create_dataset(df, img_size, augment=False, shuffle=False,
                   use_mixup=False):
    """Plain JPEG reads — no py_function, no memory leak."""
    paths  = df['path'].values
    labels = df['class'].map(class_indices).values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, l: (
            tf.cast(
                tf.image.resize(
                    tf.image.decode_jpeg(tf.io.read_file(p), channels=3),
                    img_size),
                tf.float32),
            l),
        num_parallel_calls=tf.data.AUTOTUNE)
    if augment:
        ds = ds.map(augment_train, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(
        lambda x, y: (preprocess_input(x), tf.one_hot(y, num_classes)),
        num_parallel_calls=tf.data.AUTOTUNE)
    if shuffle:
        ds = ds.shuffle(2000, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE)
    if use_mixup:
        ds = ds.map(mixup_batch, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


y_all         = train_df['class'].map(class_indices).values
cw            = compute_class_weight('balanced',
                                     classes=np.unique(y_all), y=y_all)
class_weights = dict(enumerate(cw))
print(f"Class weights: {class_weights}")


def build_model(img_size, trainable_base=False):
    base = EfficientNetB3(include_top=False, weights='imagenet',
                          input_shape=img_size + (3,), pooling='avg')
    base.trainable = trainable_base
    inp = layers.Input(shape=img_size + (3,))
    x   = base(inp, training=False)
    x   = layers.BatchNormalization()(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    x   = layers.Dense(512, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(0.5)(x)
    x   = layers.Dense(256, activation='relu',
                        kernel_regularizer=tf.keras.regularizers.l2(1e-4))(x)
    x   = layers.Dropout(DROPOUT_RATE)(x)
    out = layers.Dense(num_classes, activation='softmax',
                        dtype='float32')(x)
    return models.Model(inp, out), base


print("\n--- Quick baseline (3 epochs) ---")

b_train = train_df.groupby('class').apply(
    lambda x: x.sample(min(30, len(x)), random_state=SEED)
).reset_index(drop=True)
b_val = val_df.groupby('class').apply(
    lambda x: x.sample(min(10, len(x)), random_state=SEED)
).reset_index(drop=True)

b_tr_ds = create_dataset(b_train, IMG_SIZE_S1, shuffle=True)
b_va_ds = create_dataset(b_val,   IMG_SIZE_S1)

b_model, _ = build_model(IMG_SIZE_S1)
b_model.compile(optimizer=optimizers.Adam(1e-3),
                loss=focal_loss, metrics=['accuracy'])
b_hist = b_model.fit(b_tr_ds, validation_data=b_va_ds,
                     epochs=3, verbose=1)
b_acc  = b_hist.history['val_accuracy'][-1]
print(f"Baseline val acc: {b_acc:.1%}", end="  ")
if   b_acc < 0.30: print("⚠ BELOW RANDOM — check data")
elif b_acc < 0.50: print("⚠ LOW — pipeline ok, needs training")
else:              print("✓ Reasonable — proceed")

del b_model, b_tr_ds, b_va_ds
gc.collect()


print("\n--- Phase 1: Head training at 224×224 ---")

tr_ds_s1 = create_dataset(train_df, IMG_SIZE_S1,
                           augment=True, shuffle=True, use_mixup=True)
va_ds_s1 = create_dataset(val_df, IMG_SIZE_S1)

model, base_model = build_model(IMG_SIZE_S1)

try:
    opt_head = tf.keras.optimizers.experimental.AdamW(
        learning_rate=LR_HEAD, weight_decay=WEIGHT_DECAY)
except AttributeError:
    opt_head = optimizers.Adam(LR_HEAD)

model.compile(
    optimizer=opt_head, loss=focal_loss,
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')])

cb_p1 = [
    callbacks.ModelCheckpoint(
        os.path.join(RUN_DIR, "best_head.keras"),
        monitor='val_loss', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=5,
        restore_best_weights=True, verbose=1),
    callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.3, patience=3,
        min_lr=1e-7, verbose=1),
    callbacks.CSVLogger(os.path.join(RUN_DIR, "phase1_log.csv"))
]

hist_p1 = model.fit(
    tr_ds_s1, validation_data=va_ds_s1,
    epochs=EPOCHS_HEAD, class_weight=class_weights,
    callbacks=cb_p1, verbose=1)

del tr_ds_s1, va_ds_s1
gc.collect()


print("\n--- Phase 2: Fine-tune at 300×300 ---")

model_s2, base_s2 = build_model(IMG_SIZE_S2)

# Transfer weights from Phase 1
transferred = 0
for layer_s2 in model_s2.layers:
    try:
        w = model.get_layer(layer_s2.name).get_weights()
        if w:
            layer_s2.set_weights(w)
            transferred += 1
    except (ValueError, AttributeError):
        pass
print(f"  Weights transferred: {transferred}")

del model
gc.collect()

# Unfreeze top 60 layers, keep BN frozen
base_s2.trainable = True
for layer in base_s2.layers[:-UNFREEZE_TOP]:
    layer.trainable = False
for layer in base_s2.layers:
    if isinstance(layer, layers.BatchNormalization):
        layer.trainable = False
print(f"  Trainable layers: {sum(1 for l in model_s2.layers if l.trainable)}")

# ── Build backbone_extractor BEFORE fit() ─────────────────────────────────────
# Keras 3: EarlyStopping.restore_best_weights() severs tensor graph
# connections between the outer model scope and the nested EfficientNet
# backbone scope. Any Model() built after fit() cannot connect
# model.input -> backbone.get_layer('top_conv').output across scopes.
#
# Solution: build backbone_extractor NOW while the graph is still live.
# After fit() we copy the trained weights into it and save.
print("\nBuilding backbone_extractor before fit()...")
backbone_extractor = None
try:
    _bb  = model_s2.get_layer('efficientnetb3')
    _tc  = _bb.get_layer('top_conv')
    backbone_extractor = tf.keras.Model(
        inputs  = _bb.input,
        outputs = [_tc.output, _bb.output],
        name    = 'backbone_extractor')
    _d       = tf.zeros([1, 300, 300, 3])
    _co, _bo = backbone_extractor(_d, training=False)
    print(f"  Verified — conv: {_co.shape}  backbone: {_bo.shape}")
    del _d, _co, _bo
except Exception as e:
    print(f"  Failed: {e}")

# Build datasets
tr_ds_s2 = create_dataset(train_df, IMG_SIZE_S2,
                           augment=True, shuffle=True, use_mixup=True)
va_ds_s2 = create_dataset(val_df, IMG_SIZE_S2)

total_steps = (len(train_df) // BATCH_SIZE) * EPOCHS_FINE
lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=LR_FINE,
    decay_steps=total_steps,
    alpha=1e-7)

try:
    opt_fine = tf.keras.optimizers.experimental.AdamW(
        learning_rate=lr_schedule, weight_decay=WEIGHT_DECAY)
except AttributeError:
    opt_fine = optimizers.Adam(lr_schedule)

model_s2.compile(
    optimizer=opt_fine, loss=focal_loss,
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')])

cb_p2 = [
    callbacks.ModelCheckpoint(
        os.path.join(RUN_DIR, "best_finetune.keras"),
        monitor='val_loss', save_best_only=True, verbose=1),
    callbacks.EarlyStopping(
        monitor='val_loss', patience=EARLY_STOP_PAT,
        restore_best_weights=True, verbose=1),
    callbacks.CSVLogger(os.path.join(RUN_DIR, "phase2_log.csv"))
]

hist_p2 = model_s2.fit(
    tr_ds_s2, validation_data=va_ds_s2,
    epochs=EPOCHS_FINE, class_weight=class_weights,
    callbacks=cb_p2, verbose=1)

# Save backbone_extractor with trained weights
if backbone_extractor is not None:
    try:
        backbone_extractor.set_weights(
            model_s2.get_layer('efficientnetb3').get_weights())
        backbone_extractor.save(
            os.path.join(RUN_DIR, "backbone_extractor.keras"))
        print("✓ backbone_extractor saved")
    except Exception as e:
        print(f"  backbone_extractor save failed: {e}")

model_s2.save(os.path.join(RUN_DIR, "final_model.keras"))
print("✓ final_model saved")

del tr_ds_s2, va_ds_s2
gc.collect()

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
all_acc   = hist_p1.history['accuracy']     + hist_p2.history['accuracy']
all_vacc  = hist_p1.history['val_accuracy'] + hist_p2.history['val_accuracy']
all_loss  = hist_p1.history['loss']         + hist_p2.history['loss']
all_vloss = hist_p1.history['val_loss']     + hist_p2.history['val_loss']
ep        = range(1, len(all_acc) + 1)
unfreeze  = len(hist_p1.history['loss'])
axes[0].plot(ep, all_acc,  label='Train acc')
axes[0].plot(ep, all_vacc, label='Val acc')
axes[0].axvline(x=unfreeze, color='gray', linestyle='--', label='Unfreeze')
axes[0].set_title('Accuracy'); axes[0].legend(); axes[0].set_xlabel('Epoch')
axes[1].plot(ep, all_loss,  label='Train loss')
axes[1].plot(ep, all_vloss, label='Val loss')
axes[1].axvline(x=unfreeze, color='gray', linestyle='--')
axes[1].set_title('Loss'); axes[1].legend(); axes[1].set_xlabel('Epoch')
plt.suptitle('Training Curves — v5.1', fontsize=13)
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "training_curves.png"), dpi=150)
plt.show()

gap = all_acc[-1] - all_vacc[-1]
print(f"Train: {all_acc[-1]:.1%}  Val: {all_vacc[-1]:.1%}  Gap: {gap:.1%}")


print("\n--- Evaluation on test set ---")

best_model = tf.keras.models.load_model(
    os.path.join(RUN_DIR, "best_finetune.keras"),
    custom_objects={'FocalLoss': FocalLoss})

test_ds = create_dataset(test_df, IMG_SIZE_S2)

y_true_l, y_pred_l, y_prob_l = [], [], []
for imgs, lbls in test_ds:
    probs = best_model.predict(imgs, verbose=0)
    y_prob_l.extend(probs.tolist())
    y_pred_l.extend(np.argmax(probs, axis=1).tolist())
    y_true_l.extend(np.argmax(lbls.numpy(), axis=1).tolist())

y_true = np.array(y_true_l)
y_pred = np.array(y_pred_l)
y_prob = np.array(y_prob_l)

report = classification_report(y_true, y_pred, target_names=classes, digits=4)
print("\n" + report)
with open(os.path.join(RUN_DIR, "classification_report.txt"), 'w') as f:
    f.write(report)

print("Per-class AUROC:")
for i, cls in enumerate(classes):
    try:
        auc = roc_auc_score((y_true == i).astype(int), y_prob[:, i])
        print(f"  {cls:20s}: {auc:.4f}")
    except Exception as e:
        print(f"  {cls:20s}: N/A ({e})")

cm    = confusion_matrix(y_true, y_pred)
cm_df = pd.DataFrame(cm, index=classes, columns=classes)
cm_df.to_csv(os.path.join(RUN_DIR, "confusion_matrix.csv"))
plt.figure(figsize=(8, 6))
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues',
            linewidths=0.5, linecolor='gray')
plt.title('Confusion Matrix — v5.1')
plt.ylabel('True'); plt.xlabel('Predicted')
plt.tight_layout()
plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"), dpi=300)
plt.show()


print("\n--- GradCAM ---")

backbone_ext = tf.keras.models.load_model(
    os.path.join(RUN_DIR, "backbone_extractor.keras"))
print(f"backbone_extractor outputs: {[o.shape for o in backbone_ext.outputs]}")

# Head layers = everything in best_model that is not InputLayer or Functional
head_layers = [l for l in best_model.layers
               if l.__class__.__name__ not in ('InputLayer', 'Functional')]
print(f"Head layers: {[l.name for l in head_layers]}")


def get_gradcam_heatmap(img_array, class_idx):
    img_t = tf.cast(img_array[np.newaxis], tf.float32)
    with tf.GradientTape() as tape:
        conv_out, backbone_out = backbone_ext(img_t, training=False)
        tape.watch(conv_out)
        x = backbone_out
        for lyr in head_layers:
            try:    x = lyr(x, training=False)
            except: x = lyr(x)
        loss = x[:, class_idx]
    grads = tape.gradient(loss, conv_out)
    if grads is None:
        return np.zeros(conv_out.shape[1:3])
    pooled  = tf.reduce_mean(grads, axis=(0, 1, 2))
    heatmap = tf.squeeze(conv_out[0] @ pooled[..., tf.newaxis]).numpy()
    heatmap = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap


def overlay_heatmap(img_rgb, heatmap, alpha=0.4):
    h, w    = img_rgb.shape[:2]
    hmap_r  = cv2.resize(heatmap, (w, h))
    jet     = cv2.applyColorMap(np.uint8(255 * hmap_r), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_rgb, 1 - alpha,
                              cv2.cvtColor(jet, cv2.COLOR_BGR2RGB), alpha, 0)
    return overlay, hmap_r


def iou_score(heatmap, leaf_mask, threshold=0.2):
    h, w     = leaf_mask.shape
    hmap     = cv2.resize(heatmap, (w, h))
    hmap_bin = (hmap >= threshold).astype(np.uint8)
    leaf_bin = (leaf_mask > 0).astype(np.uint8)
    inter    = (hmap_bin & leaf_bin).sum()
    union    = (hmap_bin | leaf_bin).sum()
    return float(inter / union) if union > 0 else 0.0


# Sanity check
test_paths = test_df['path'].values
_bgr = cv2.resize(cv2.imread(test_paths[0]), IMG_SIZE_S2[::-1])
_pre = preprocess_input(
    cv2.cvtColor(_bgr, cv2.COLOR_BGR2RGB).astype(np.float32))
_hm  = get_gradcam_heatmap(_pre, 0)
print(f"Sanity check: shape={_hm.shape}  max={_hm.max():.3f}")
print("  ✓ GradCAM working" if _hm.max() > 0 else "  ⚠ Still zero")

# 8 images per class
gradcam_stats = []
for cls_name in classes:
    correct = [i for i, (t, p) in enumerate(zip(y_true, y_pred))
               if classes[t] == cls_name and t == p]
    wrong   = [i for i, (t, p) in enumerate(zip(y_true, y_pred))
               if classes[t] == cls_name and t != p]
    pool    = (correct[:4] + wrong[:4])[:8]
    if not pool:
        continue

    n = len(pool)
    fig, axes = plt.subplots(n, 3, figsize=(15, 4 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for ri, gidx in enumerate(pool):
        img_bgr = cv2.resize(cv2.imread(test_paths[gidx]), IMG_SIZE_S2[::-1])
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        # Leaf mask for IoU scoring
        hsv   = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
        lmask = cv2.inRange(hsv, np.array([25, 25, 25]),
                            np.array([90, 255, 255]))
        if float((lmask > 0).sum()) / lmask.size > 0.85:
            gray  = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
            _, lmask = cv2.threshold(gray, 0, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        img_pre         = preprocess_input(img_rgb.astype(np.float32))
        pred_idx        = y_pred[gidx]
        heatmap         = get_gradcam_heatmap(img_pre, pred_idx)
        overlay, hmap_r = overlay_heatmap(img_rgb, heatmap)
        score           = iou_score(hmap_r, lmask)

        gradcam_stats.append({
            'class': cls_name, 'pred': classes[pred_idx],
            'correct': (y_true[gidx] == pred_idx),
            'iou_score': score,
            'confidence': float(y_prob[gidx, pred_idx])
        })

        flag  = ("✓ LEAF" if score >= 0.5 else
                 "⚠ WEAK" if score >= 0.3 else "⚠ SHORTCUT")
        c_str = "CORRECT" if y_true[gidx] == pred_idx else "WRONG"

        axes[ri, 0].imshow(img_rgb)
        axes[ri, 0].set_title(
            f"True:{cls_name}\nPred:{classes[pred_idx]} "
            f"({y_prob[gidx, pred_idx]:.1%}) [{c_str}]", fontsize=8)
        axes[ri, 0].axis('off')
        axes[ri, 1].imshow(overlay)
        axes[ri, 1].set_title(f"GradCAM\nIoU={score:.2f} {flag}", fontsize=8)
        axes[ri, 1].axis('off')
        axes[ri, 2].imshow(hmap_r, cmap='jet')
        axes[ri, 2].set_title("Raw heatmap\n(red=high activation)",
                               fontsize=8)
        axes[ri, 2].axis('off')

    plt.suptitle(
        f"GradCAM — {cls_name}\n"
        "IoU>0.5: leaf focus  |  IoU<0.3: background shortcut",
        fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(RUN_DIR,
                f"gradcam_{cls_name.replace(' ', '_')}.png"), dpi=150)
    plt.show()
    gc.collect()

gc_df = pd.DataFrame(gradcam_stats)
print("\nGradCAM IoU by class:")
print(gc_df.groupby('class')['iou_score'].describe().round(3).to_string())
gc_df.to_csv(os.path.join(RUN_DIR, "gradcam_iou_stats.csv"), index=False)
low  = gc_df[gc_df['iou_score'] < 0.3]
good = gc_df[gc_df['iou_score'] >= 0.5]
print(f"\n  Shortcut (IoU<0.3) : {len(low)}/{len(gc_df)}")
print(f"  Leaf focus (IoU≥0.5): {len(good)}/{len(gc_df)}")


print("\n--- Robustness suite ---")

bg_pool_test = []
for cls in classes:
    imgs = [f for f in os.listdir(class_dirs[cls])
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    for fname in imgs[:8]:
        img = cv2.imread(os.path.join(class_dirs[cls], fname))
        if img is None:
            continue
        img = cv2.resize(img, (300, 300))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        fg  = cv2.inRange(hsv, np.array([25, 25, 25]),
                          np.array([90, 255, 255]))
        p   = img.copy()
        p[fg > 0] = 128
        bg_pool_test.append(p)

test_paths_raw  = test_df['path'].values
test_labels_raw = test_df['class'].map(class_indices).values


def eval_scenario(transform_fn, desc, n=50):
    idxs  = np.random.choice(len(test_paths_raw),
                              min(n, len(test_paths_raw)), replace=False)
    preds, trues = [], []
    for i in idxs:
        img = cv2.imread(test_paths_raw[i])
        if img is None:
            continue
        img = cv2.resize(img, IMG_SIZE_S2[::-1])
        img = transform_fn(img)
        pre = preprocess_input(
            cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32))
        prob = best_model.predict(pre[np.newaxis], verbose=0)[0]
        preds.append(np.argmax(prob))
        trues.append(test_labels_raw[i])
    acc = np.mean(np.array(preds) == np.array(trues))
    f1v = f1_score(trues, preds, average='macro', zero_division=0)
    print(f"  {desc:45s}: Acc={acc:.1%}  MacroF1={f1v:.3f}")
    return {'scenario': desc, 'accuracy': acc, 'macro_f1': f1v}


def swap_bg_fn(img):
    donor   = bg_pool_test[np.random.randint(len(bg_pool_test))]
    donor_r = cv2.resize(donor, (img.shape[1], img.shape[0]))
    gray_px = np.all(img == 128, axis=-1)
    r = img.copy()
    r[gray_px] = donor_r[gray_px]
    return r


results = []
results.append(eval_scenario(
    lambda x: x,                "S1: Original (baseline)"))
results.append(eval_scenario(
    swap_bg_fn,                 "S2: Background swapped (shortcut test)"))
results.append(eval_scenario(
    lambda x: np.clip(x.astype(np.float32) * 0.7, 0, 255).astype(np.uint8),
                                "S3: Brightness -30%"))
results.append(eval_scenario(
    lambda x: cv2.rotate(x, cv2.ROTATE_90_CLOCKWISE),
                                "S4: 90° rotation"))

rob_df = pd.DataFrame(results)
rob_df.to_csv(os.path.join(RUN_DIR, "robustness_results.csv"), index=False)

s1_acc  = rob_df[rob_df['scenario'].str.startswith('S1')]['accuracy'].values[0]
s2_acc  = rob_df[rob_df['scenario'].str.startswith('S2')]['accuracy'].values[0]
bg_drop = s1_acc - s2_acc
print(f"\nBackground-swap drop: {bg_drop:.1%}  (v5.0 was 20.8%)")
if   bg_drop < 0.08: print("  ✓ LOW — shortcut resolved")
elif bg_drop < 0.15: print("  ⚠ MODERATE — improved")
else:                print("  ⚠ HIGH — shortcut persists")


final_acc = float(np.mean(y_true == y_pred))
macro_f1  = float(f1_score(y_true, y_pred, average='macro', zero_division=0))
mean_iou  = float(gc_df['iou_score'].mean()) if len(gc_df) > 0 else 0.0

summary = {
    'version': 'v5.1', 'run_dir': RUN_DIR,
    'accuracy': final_acc, 'macro_f1': macro_f1,
    'mean_gradcam_iou': mean_iou, 'bg_swap_drop': bg_drop,
    'class_indices': class_indices,
    'artifacts': {
        'best_model':         os.path.join(RUN_DIR, "best_finetune.keras"),
        'backbone_extractor': os.path.join(RUN_DIR, "backbone_extractor.keras"),
        'class_indices':      os.path.join(RUN_DIR, "class_indices.json"),
    }
}
with open(os.path.join(RUN_DIR, "run_summary.json"), 'w') as f:
    json.dump({k: str(v) for k, v in summary.items()}, f, indent=2)

print(f"""
{'='*60}
v5.1 COMPLETE
  Accuracy           : {final_acc:.1%}
  Macro F1           : {macro_f1:.3f}
  Mean GradCAM IoU   : {mean_iou:.3f}
  BG-swap drop       : {bg_drop:.1%}  (v5.0 was 20.8%)

  best_finetune.keras       — classification model
  backbone_extractor.keras  — GradCAM frontend
  class_indices.json        — label mapping

  Run dir: {RUN_DIR}
{'='*60}
""")
