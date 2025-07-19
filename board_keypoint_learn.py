import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import cv2
import argparse

import keypoint_utils as kpu

CHECKPOINT_PATH = "board_keypoint_detector.h5"
IMG_SIZE = kpu.IMG_SIZE
HM_SIZE = kpu.HM_SIZE
NUM_CORNERS = kpu.NUM_CORNERS

# =========================
# AUGMENTATION FUNCTIONS
# =========================

def identity(img, kps):
    return img, kps

def rotate_90(img, kps):
    img_rot = np.ascontiguousarray(np.rot90(img, 1))
    # (x, y) -> (y, W-1-x)
    kps_rot = np.stack([kps[:, 1], IMG_SIZE - 1 - kps[:, 0]], axis=-1)
    return img_rot, kps_rot

def rotate_180(img, kps):
    img_rot = np.ascontiguousarray(np.rot90(img, 2))
    # (x, y) -> (W-1-x, H-1-y)
    kps_rot = np.stack([IMG_SIZE - 1 - kps[:, 0], IMG_SIZE - 1 - kps[:, 1]], axis=-1)
    return img_rot, kps_rot

def rotate_neg90(img, kps):
    img_rot = np.ascontiguousarray(np.rot90(img, -1))
    # (x, y) -> (H-1-y, x)
    kps_rot = np.stack([IMG_SIZE - 1 - kps[:, 1], kps[:, 0]], axis=-1)
    return img_rot, kps_rot

AUGMENTATIONS = [
    identity,
    rotate_90,
    rotate_180,
    rotate_neg90,
]

# =========================
# DATASET WITH AUGMENTATION
# =========================

class BoardDataset(tf.keras.utils.Sequence):
    def __init__(self, img_paths, kps, batch_size=8, augmentations=None):
        self.orig_img_paths = img_paths
        self.orig_kps = kps
        self.batch_size = batch_size
        self.augmentations = augmentations if augmentations is not None else [identity]

        # Precompute augmented dataset (img, keypoints)
        self.aug_img_kps = []
        for path, kp in zip(self.orig_img_paths, self.orig_kps):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, kp, IMG_SIZE)
            for aug_fn in self.augmentations:
                aug_img, aug_kps = aug_fn(img_resized.copy(), kps_rescaled.copy())
                self.aug_img_kps.append((aug_img, aug_kps))
        self.indices = np.arange(len(self.aug_img_kps))

    def __len__(self):
        return int(np.ceil(len(self.aug_img_kps) / self.batch_size))

    def __getitem__(self, idx):
        batch = self.aug_img_kps[idx * self.batch_size:(idx + 1) * self.batch_size]
        imgs, hms, offs, masks = [], [], [], []
        for img, kp in batch:
            imgs.append(img / 255.0)
            hm, off, mask = kpu.generate_targets(kp)
            hms.append(hm)
            offs.append(off)
            masks.append(mask)
        return np.array(imgs), {'heatmap': np.array(hms), 'offsets': np.array(offs), 'mask': np.array(masks)}

    def single_samples(self):
        # For previewing all examples one by one
        for img, kp in self.aug_img_kps:
            hm, off, mask = kpu.generate_targets(kp)
            yield img, kp, hm, off, mask

# =========================
# MODEL
# =========================

def build_model():
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 7, strides=2, padding='same', activation='relu')(inputs)
    x = layers.Conv2D(64, 5, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(128, 3, strides=2, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    x = layers.Conv2D(256, 3, strides=1, padding='same', activation='relu')(x)
    heatmap = layers.Conv2D(1, 1, activation='sigmoid', name='heatmap')(x)
    offsets = layers.Conv2D(2, 1, activation=None, name='offsets')(x)
    return keras.Model(inputs, [heatmap, offsets])

# =========================
# LOSSES
# =========================

def heatmap_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def offset_loss(y_true, y_pred, mask):
    diff = (y_true - y_pred) * mask
    return tf.reduce_sum(tf.abs(diff)) / (tf.reduce_sum(mask) + 1e-6)

# =========================
# TRAINING LOOP
# =========================

def train(model, dataset, epochs=15, checkpoint_dir="./checkpoints"):
    optimizer = keras.optimizers.Adam(1e-3)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    for epoch in range(epochs):
        hm_losses, off_losses = [], []
        for imgs, targets in dataset:
            hms_true = targets['heatmap']
            offs_true = targets['offsets']
            mask = targets['mask']
            with tf.GradientTape() as tape:
                hms_pred, offs_pred = model(imgs, training=True)
                l1 = heatmap_loss(hms_true, hms_pred)
                l2 = 0.01 * offset_loss(offs_true, offs_pred, mask)
                loss = l1 + l2
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            hm_losses.append(l1.numpy())
            off_losses.append(l2.numpy())

        mean_hm_loss = np.mean(hm_losses)
        mean_off_loss = np.mean(off_losses)
        mean_loss = mean_hm_loss + mean_off_loss

        print(f"Epoch {epoch+1}/{epochs}: hm_loss={mean_hm_loss:.4f}, off_loss={mean_off_loss:.4f}, total_loss={mean_loss:.4f}")

        # Save checkpoint every epoch
        # model.save_weights(os.path.join(checkpoint_dir, f"epoch_{epoch+1:03d}.h5"))

        # Optionally, only keep best checkpoint
        if mean_loss < best_loss:
            best_loss = mean_loss
            model.save(os.path.join(checkpoint_dir, "best.h5"))

    return model

# =========================
# EVALUATION
# =========================

def infer_and_show(model, img_path, kp_true):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, kp_true, IMG_SIZE)
    inp = np.expand_dims(img_resized / 255.0, 0)
    hm_pred, off_pred = model.predict(inp)
    hm_pred = hm_pred[0]
    off_pred = off_pred[0]

    # reconstruct from heatmap & offsets
    hm_exp = np.expand_dims(hm_pred[...,0], -1)
    rec_kps = kpu.reconstruct_keypoints(hm_exp, off_pred, threshold=0.5)

    img_show = img_resized.copy()
    # draw up to NUM_CORNERS strongest
    if len(rec_kps) > NUM_CORNERS:
        flat = hm_pred[...,0].flatten()
        idxs = np.argpartition(-flat, NUM_CORNERS)[:NUM_CORNERS]
        ys, xs = np.unravel_index(idxs, hm_pred[...,0].shape)
        rec_kps = [((x+off_pred[y,x,0])/(HM_SIZE/IMG_SIZE), (y+off_pred[y,x,1])/(HM_SIZE/IMG_SIZE)) for y,x in zip(ys, xs)]

    for rx, ry in rec_kps:
        cv2.circle(img_show, (int(rx), int(ry)), 8, (0,255,0), 2)
    for kp in kps_rescaled:
        cv2.drawMarker(img_show, (int(kp[0]), int(kp[1])), (255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)

    import matplotlib.pyplot as plt
    plt.imshow(img_show)
    plt.title("Green: Predicted, Blue: GT (rescaled)")
    plt.show()

def show_heatmap_comparison(img_path, keypoints, model):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, keypoints, IMG_SIZE)
    inp = np.expand_dims(img_resized / 255.0, 0)
    hm_pred, _ = model.predict(inp)
    hm_pred = hm_pred[0,...,0]
    hm_gt, _, _ = kpu.generate_targets(kps_rescaled)

    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img_resized)
    axes[0].set_title("Input Image")
    axes[1].imshow(hm_gt[...,0], cmap='hot')
    axes[1].set_title("GT Heatmap")
    axes[2].imshow(hm_pred, cmap='hot')
    axes[2].set_title("Predicted Heatmap")
    plt.show()

# =========================
# PREVIEW AUGMENTED DATA (OPENCV)
# =========================

def preview_augmented_dataset(dataset):
    print("Previewing augmented dataset (ESC to quit, any key for next)...")
    for img, kps, hm, off, mask in dataset.single_samples():
        img_disp = img.astype(np.uint8).copy()
        for kp in kps:
            cv2.drawMarker(img_disp, (int(kp[0]), int(kp[1])), (0,255,0), markerType=cv2.MARKER_CROSS, thickness=2, line_type=cv2.LINE_AA)

        # Make heatmap 3-channel and resize to match image
        heat_disp = cv2.normalize(hm[...,0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_disp = cv2.applyColorMap(heat_disp, cv2.COLORMAP_JET)
        heat_disp = cv2.resize(heat_disp, (img_disp.shape[1], img_disp.shape[0]), interpolation=cv2.INTER_NEAREST)

        stacked = np.hstack([cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR), heat_disp])

        cv2.imshow("Image (left) + Heatmap (right). ESC to quit, any key for next.", stacked)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()

# =========================
# MAIN SCRIPT
# =========================

def main():
    parser = argparse.ArgumentParser(description='Board keypoint detector training and evaluation.')
    parser.add_argument('--annotations', type=str, default='annotations.txt')
    parser.add_argument('--images', type=str, default='files')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--detections', type=int, default=3)
    args = parser.parse_args()

    img_paths, keypoints = kpu.parse_annotations(args.annotations, args.images)
    ds = BoardDataset(img_paths, keypoints, batch_size=args.batch_size, augmentations=AUGMENTATIONS)

    # ==== Preview a few augmented examples before training ====
    preview_augmented_dataset(ds)

    # ==== Build model ====
    model = build_model()

    # ==== Load checkpoint if exists ====
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_weights(CHECKPOINT_PATH)
            print(f"Loaded existing weights from {CHECKPOINT_PATH}, continuing training.")
        except Exception as e:
            print(f"Error loading weights, starting from scratch: {e}")
    else:
        print("No checkpoint found, training from scratch.")

    # ==== Train ====
    model = train(model, ds, epochs=args.epochs)

    # ==== Save weights ====
    model.save(CHECKPOINT_PATH)
    print(f"Model weights saved to {CHECKPOINT_PATH}")

    # ==== Evaluate ====
    show_indices = random.sample(range(len(img_paths)), min(args.detections, len(img_paths)))
    for idx in show_indices:
        print(f"Detection on: {img_paths[idx]}")
        infer_and_show(model, img_paths[idx], keypoints[idx])
        show_heatmap_comparison(img_paths[idx], keypoints[idx], model)

if __name__ == "__main__":
    main()
