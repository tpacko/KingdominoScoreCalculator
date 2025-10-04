import os
from time import sleep

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random
import cv2
import argparse

from tqdm import tqdm
from sklearn.model_selection import train_test_split

import keypoint_utils as kpu


# ============= CONFIGURATION =============

CHECKPOINT_PATH = "board_keypoint_detector.h5"
IMG_SIZE = kpu.IMG_SIZE
HM_SIZE = kpu.HM_SIZE
NUM_CORNERS = kpu.NUM_CORNERS

ANNOTATIONS_PATH = 'annotations.txt'
IMAGES_PATH = 'files'
EPOCHS = 15
BATCH_SIZE = 4
DETECTIONS = 5
LEARNING_RATE = 1e-4
# ============= END CONFIGURATION =============

# ============= Augmentations (same as above) =============

def identity(img, kps):
    return img, kps

def rotate_90(img, kps):
    img_rot = np.ascontiguousarray(np.rot90(img, 1))
    kps_rot = np.stack([kps[:, 1], IMG_SIZE - 1 - kps[:, 0]], axis=-1)
    return img_rot, kps_rot

def rotate_180(img, kps):
    img_rot = np.ascontiguousarray(np.rot90(img, 2))
    kps_rot = np.stack([IMG_SIZE - 1 - kps[:, 0], IMG_SIZE - 1 - kps[:, 1]], axis=-1)
    return img_rot, kps_rot

def rotate_neg90(img, kps):
    img_rot = np.ascontiguousarray(np.rot90(img, -1))
    kps_rot = np.stack([IMG_SIZE - 1 - kps[:, 1], kps[:, 0]], axis=-1)
    return img_rot, kps_rot

def keypoints_in_bounds(kps, size):
    return not (np.any(kps < 0) or np.any(kps >= size))

def horizontal_flip(img, kps):
    img_flipped = np.fliplr(img)
    kps_flipped = kps.copy()
    kps_flipped[:, 0] = IMG_SIZE - 1 - kps[:, 0]
    if not keypoints_in_bounds(kps_flipped, IMG_SIZE):
        return img, kps
    return img_flipped, kps_flipped

def vertical_flip(img, kps):
    img_flipped = np.flipud(img)
    kps_flipped = kps.copy()
    kps_flipped[:, 1] = IMG_SIZE - 1 - kps[:, 1]
    if not keypoints_in_bounds(kps_flipped, IMG_SIZE):
        return img, kps
    return img_flipped, kps_flipped

def random_small_rotation(img, kps, angle_range=30):
    angle = np.random.uniform(-angle_range, angle_range)
    center = (IMG_SIZE/2, IMG_SIZE/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    ones = np.ones((kps.shape[0], 1))
    kps_homo = np.concatenate([kps, ones], axis=1)
    kps_rot = (M @ kps_homo.T).T
    if not keypoints_in_bounds(kps_rot, IMG_SIZE):
        return img, kps
    return img_rot, kps_rot

def random_scale(img, kps, scale_range=(0.9, 1.2)):
    scale = np.random.uniform(*scale_range)
    M = np.array([
        [scale, 0, (1-scale)*IMG_SIZE/2],
        [0, scale, (1-scale)*IMG_SIZE/2]
    ])
    img_scaled = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    ones = np.ones((kps.shape[0], 1))
    kps_homo = np.concatenate([kps, ones], axis=1)
    kps_scaled = (M @ kps_homo.T).T
    if not keypoints_in_bounds(kps_scaled, IMG_SIZE):
        return img, kps
    return img_scaled, kps_scaled

def random_translate(img, kps, frac=0.1):
    tx = np.random.uniform(-IMG_SIZE*frac, IMG_SIZE*frac)
    ty = np.random.uniform(-IMG_SIZE*frac, IMG_SIZE*frac)
    M = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ])
    img_trans = cv2.warpAffine(img, M, (IMG_SIZE, IMG_SIZE), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    kps_trans = kps + np.array([tx, ty])
    if not keypoints_in_bounds(kps_trans, IMG_SIZE):
        return img, kps
    return img_trans, kps_trans

def color_jitter(img, kps, brightness=0.5, contrast=0.5, saturation=0.5):
    img_out = img.astype(np.float32)
    img_out += np.random.uniform(-brightness*255, brightness*255)
    factor = np.random.uniform(1-contrast, 1+contrast)
    img_out = 127.5 + factor * (img_out - 127.5)
    img_hsv = cv2.cvtColor(np.clip(img_out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat_factor = np.random.uniform(1-saturation, 1+saturation)
    img_hsv[...,1] = np.clip(img_hsv[...,1]*sat_factor, 0, 255)
    img_out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_out.astype(np.uint8), kps

def random_perspective(img, kps, max_warp=0.05):
    margin = IMG_SIZE * max_warp
    src = np.array([
        [0, 0],
        [IMG_SIZE-1, 0],
        [IMG_SIZE-1, IMG_SIZE-1],
        [0, IMG_SIZE-1]
    ], dtype=np.float32)
    dst = src + np.random.uniform(-margin, margin, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img_warp = cv2.warpPerspective(img, M, (IMG_SIZE, IMG_SIZE), borderMode=cv2.BORDER_REFLECT_101)
    kps_homo = np.concatenate([kps, np.ones((kps.shape[0], 1))], axis=1)
    kps_warp = (M @ kps_homo.T).T
    kps_warp = kps_warp[:, :2] / (kps_warp[:, 2:]+1e-8)
    if not keypoints_in_bounds(kps_warp, IMG_SIZE):
        return img, kps
    return img_warp, kps_warp

AUGMENTATIONS = [
    identity,
    rotate_90,
    rotate_180,
    rotate_neg90,
    horizontal_flip,
    vertical_flip,
    random_small_rotation,
    random_scale,
    random_translate,
    color_jitter,
    random_perspective,
]

# ============= Dataset =============

class BoardDataset(tf.keras.utils.Sequence):
    def __init__(self, img_paths, kps, batch_size=8, augmentations=None, num_iterations=4):
        self.orig_img_paths = img_paths
        self.orig_kps = kps
        self.batch_size = batch_size
        self.augmentations = augmentations if augmentations is not None else [identity]
        self.num_iterations = num_iterations

        self.aug_img_kps = []
        for path, kp in zip(self.orig_img_paths, self.orig_kps):
            img = cv2.imread(path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, kp, IMG_SIZE)
            self.aug_img_kps.append((img_resized, kps_rescaled))
        print(f"Base dataset size: {len(self.aug_img_kps)}")

        for it in range(self.num_iterations):
            new_aug_img_kps = []
            for img, kp in self.aug_img_kps:
                aug_fn = random.choice(self.augmentations)
                aug_img, aug_kps = aug_fn(img.copy(), kp.copy())
                new_aug_img_kps.append((aug_img, aug_kps))
            self.aug_img_kps.extend(new_aug_img_kps)
            print(f"After augmentation iteration {it+1}: dataset size = {len(self.aug_img_kps)}")

        self.indices = np.arange(len(self.aug_img_kps))

    def __len__(self):
        return int(np.ceil(len(self.aug_img_kps) / self.batch_size))

    def __getitem__(self, idx):
        # Use shuffled indices for batch selection
        batch_indices = self.indices[idx * self.batch_size:(idx + 1) * self.batch_size]
        batch = [self.aug_img_kps[i] for i in batch_indices]
        imgs, hms, offs, masks, segs = [], [], [], [], []
        for img, kp in batch:
            imgs.append(img / 255.0)
            hm, off, mask, seg = kpu.generate_targets(kp)
            hms.append(hm); offs.append(off); masks.append(mask); segs.append(seg)
        return np.array(imgs), {
            'heatmap': np.array(hms),
            'offsets': np.array(offs),
            'mask':    np.array(masks),
            'segmask': np.array(segs),
        }

    def single_samples(self):
        for img, kp in random.sample(self.aug_img_kps, len(self.aug_img_kps)):
            hm, off, mask, seg = kpu.generate_targets(kp)
            yield img, kp, hm, off, mask, seg

    def on_epoch_end(self):
        np.random.shuffle(self.indices)

# ============= Model =============

def build_model(base_filters=32):
    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))

    x = layers.Conv2D(base_filters, 5, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(inputs)
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    x = layers.Conv2D(base_filters * 2, 3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Conv2D(base_filters * 2, 3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Conv2D(base_filters * 2, 3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    x = layers.Conv2D(base_filters * 4, 3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Conv2D(base_filters * 4, 3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Conv2D(base_filters * 4, 3, strides=2, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    x = layers.Conv2D(base_filters * 8, 3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Conv2D(base_filters * 8, 3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.Conv2D(base_filters * 8, 3, strides=1, padding='same', activation='relu',
                      kernel_regularizer=keras.regularizers.l2(1e-4))(x)

    heatmap = layers.Conv2D(1, 1, activation='sigmoid', name='heatmap')(x)
    offsets = layers.Conv2D(2, 1, activation='tanh', name='offsets')(x)
    segmask = layers.Conv2D(1, 1, activation='sigmoid', name='segmentation')(x)
    return keras.Model(inputs, [heatmap, offsets, segmask])

# ============= Losses =============

def sampled_bce_loss(y_true, y_pred, num_negatives=200):
    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])
    bce = tf.keras.backend.binary_crossentropy(y_true, y_pred)
    bce = tf.reshape(bce, [-1])
    pos_idx = tf.where(y_true > 0.5)[:, 0]
    neg_idx = tf.where(y_true <= 0.5)[:, 0]
    pos_loss = tf.gather(bce, pos_idx)
    n_neg = tf.shape(neg_idx)[0]
    k = tf.minimum(num_negatives, n_neg)
    rand_scores = tf.random.uniform([n_neg], dtype=tf.float32)
    sample_pos_in_neg = tf.math.top_k(rand_scores, k, sorted=False).indices
    sampled_neg_idx = tf.gather(neg_idx, sample_pos_in_neg)
    sampled_neg_idx = tf.stop_gradient(sampled_neg_idx)
    neg_loss = tf.gather(bce, sampled_neg_idx)
    parts = []
    if pos_loss.shape.rank is not None:
        parts.append(pos_loss)
    else:
        parts.append(tf.zeros([0], bce.dtype))
    parts.append(neg_loss)
    total = tf.concat(parts, axis=0)
    return tf.reduce_mean(total)

def bce_loss(y_true, y_pred):
    return tf.reduce_mean(tf.keras.losses.binary_crossentropy(y_true, y_pred))

def tversky_loss(y_true, y_pred, alpha=0.4, beta=0.6, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    tp = tf.reduce_sum(y_true_f * y_pred_f)
    fn = tf.reduce_sum(y_true_f * (1 - y_pred_f))
    fp = tf.reduce_sum((1 - y_true_f) * y_pred_f)
    tversky = (tp + smooth) / (tp + alpha * fn + beta * fp + smooth)
    return 1 - tversky

def focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    bce_exp = tf.exp(-bce)
    focal = alpha * (1 - bce_exp) ** gamma * bce
    return tf.reduce_mean(focal)

def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice

def offset_loss(y_true, y_pred, mask):
    diff = (y_true - y_pred) * mask
    return tf.reduce_sum(tf.abs(diff)) / (tf.reduce_sum(mask) + 1e-6)

def seg_loss(y_true, y_pred, w_bce=1.0, w_dice=1.0):
    return w_bce * bce_loss(y_true, y_pred) + w_dice * dice_loss(y_true, y_pred)

# ============= Training =============

def train(model, train_dataset, val_dataset, epochs=15, checkpoint_dir="./checkpoints"):
    optimizer = keras.optimizers.Adam(LEARNING_RATE)
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_loss = float("inf")

    def l1_loss(y_true, y_pred):  # heatmap
        return 100*bce_loss(y_true, y_pred)

    def l2_loss(offs_true, offs_pred, mask):  # offsets
        return 0.0 * offset_loss(offs_true, offs_pred, mask)

    def l3_loss(seg_true, seg_pred):  # segmentation
        return seg_loss(seg_true, seg_pred, w_bce=1.0, w_dice=1.0)

    train_losses = []
    val_losses = []
    import matplotlib.pyplot as plt
    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))
    for epoch in range(epochs):
        train_dataset.on_epoch_end()
        val_dataset.on_epoch_end()

        hm_losses, off_losses, seg_losses = [], [], []
        print(f"\nEpoch {epoch+1}/{epochs}")
        sleep(1)

        for imgs, targets in tqdm(train_dataset, desc='Training', leave=True):
            hms_true = targets['heatmap']
            offs_true = targets['offsets']
            mask      = targets['mask']
            seg_true  = targets['segmask']
            with tf.GradientTape() as tape:
                hms_pred, offs_pred, seg_pred = model(imgs, training=True)
                l1 = l1_loss(hms_true, hms_pred)
                l2 = l2_loss(offs_true, offs_pred, mask)
                l3 = l3_loss(seg_true, seg_pred)
                loss = l1 + l2 + l3
            grads = tape.gradient(loss, model.trainable_weights)
            optimizer.apply_gradients(zip(grads, model.trainable_weights))
            hm_losses.append(float(l1)); off_losses.append(float(l2)); seg_losses.append(float(l3))
        mean_hm_loss = np.mean(hm_losses)
        mean_off_loss = np.mean(off_losses)
        mean_seg_loss = np.mean(seg_losses)
        mean_loss = mean_hm_loss + mean_off_loss + mean_seg_loss
        train_losses.append(mean_loss)

        val_hm_losses, val_off_losses, val_seg_losses = [], [], []
        for imgs, targets in tqdm(val_dataset, desc='Validation', leave=True):
            hms_true = targets['heatmap']
            offs_true = targets['offsets']
            mask      = targets['mask']
            seg_true  = targets['segmask']
            hms_pred, offs_pred, seg_pred = model(imgs, training=False)
            l1 = l1_loss(hms_true, hms_pred)
            l2 = l2_loss(offs_true, offs_pred, mask)
            l3 = l3_loss(seg_true, seg_pred)
            val_hm_losses.append(float(l1)); val_off_losses.append(float(l2)); val_seg_losses.append(float(l3))
        mean_val_hm_loss = np.mean(val_hm_losses)
        mean_val_off_loss = np.mean(val_off_losses)
        mean_val_seg_loss = np.mean(val_seg_losses)
        mean_val_loss = mean_val_hm_loss + mean_val_off_loss + mean_val_seg_loss
        val_losses.append(mean_val_loss)

        ax.clear()
        ax.plot(train_losses, label='Train Loss')
        ax.plot(val_losses, label='Validation Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Training and Validation Loss')
        ax.legend()
        ax.grid(True)
        fig.tight_layout()
        plt.draw()
        plt.pause(0.1)

        print(f"Train: hm_loss={mean_hm_loss:.4f}, off_loss={mean_off_loss:.4f}, seg_loss={mean_seg_loss:.4f}, total_loss={mean_loss:.4f} | "
              f"Val: hm_loss={mean_val_hm_loss:.4f}, off_loss={mean_val_off_loss:.4f}, seg_loss={mean_val_seg_loss:.4f}, total_loss={mean_val_loss:.4f}")

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            print(f"New best validation loss: {best_loss:.4f}, saving model weights...")
            model.save(os.path.join(checkpoint_dir, "best.h5"))

        sleep(5)
    plt.ioff()
    plt.show()
    return model

# ============= Evaluation =============

def infer_and_show(model, img_path, kp_true):
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, kp_true, IMG_SIZE)
    inp = np.expand_dims(img_resized / 255.0, 0)
    hm_pred, off_pred, seg_pred = model.predict(inp)
    hm_pred = hm_pred[0]
    off_pred = off_pred[0]
    seg_pred = seg_pred[0, ..., 0]

    hm_exp = np.expand_dims(hm_pred[...,0], -1)
    rec_kps = kpu.reconstruct_keypoints(hm_exp, off_pred, threshold=0.5)

    img_show = img_resized.copy()
    if len(rec_kps) > NUM_CORNERS:
        flat = hm_pred[...,0].flatten()
        idxs = np.argpartition(-flat, NUM_CORNERS)[:NUM_CORNERS]
        ys, xs = np.unravel_index(idxs, hm_pred[...,0].shape)
        rec_kps = [((x+off_pred[y,x,0])/(HM_SIZE/IMG_SIZE), (y+off_pred[y,x,1])/(HM_SIZE/IMG_SIZE)) for y,x in zip(ys, xs)]

    for rx, ry in rec_kps:
        cv2.circle(img_show, (int(rx), int(ry)), 8, (0,255,0), 2)
    for kp in kps_rescaled:
        cv2.drawMarker(img_show, (int(kp[0]), int(kp[1])), (255,0,0), markerType=cv2.MARKER_CROSS, thickness=2)

    seg_up = cv2.resize(seg_pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    seg_color = cv2.applyColorMap((seg_up*255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_show, 0.7, seg_color, 0.3, 0)

    import matplotlib.pyplot as plt
    plt.imshow(overlay)
    plt.title("Green: Predicted, Blue: GT (rescaled) | Seg overlay")
    plt.show()

def show_heatmap_comparison(img_path, keypoints, model):
    import matplotlib.pyplot as plt

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, keypoints, IMG_SIZE)

    inp = np.expand_dims(img_resized / 255.0, 0)
    hm_pred, _, seg_pred = model.predict(inp)
    hm_pred = hm_pred[0, ..., 0]
    seg_pred = seg_pred[0, ..., 0]

    hm_gt, _, _, seg_gt = kpu.generate_targets(kps_rescaled)
    hm_gt = hm_gt[..., 0]
    seg_gt = seg_gt[..., 0]

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    axes[0].imshow(img_resized); axes[0].set_title("Input"); axes[0].axis('off')
    axes[1].imshow(hm_gt, cmap='hot'); axes[1].set_title("GT Heatmap"); axes[1].axis('off')
    axes[2].imshow(hm_pred, cmap='hot'); axes[2].set_title("Pred Heatmap"); axes[2].axis('off')
    axes[3].imshow(seg_gt, cmap='gray'); axes[3].set_title("GT Seg"); axes[3].axis('off')
    axes[4].imshow(seg_pred, cmap='jet'); axes[4].set_title("Pred Seg"); axes[4].axis('off')
    plt.tight_layout()
    plt.show()

# ============= Preview Augmented Dataset =============

def preview_augmented_dataset(dataset):
    print("Previewing augmented dataset (ESC to quit, any key for next)...")
    for img, kps, hm, off, mask, seg in dataset.single_samples():
        img_disp = img.astype(np.uint8).copy()
        for kp in kps:
            cv2.drawMarker(img_disp, (int(kp[0]), int(kp[1])), (0,255,0), markerType=cv2.MARKER_CROSS, thickness=2, line_type=cv2.LINE_AA)

        heat_disp = cv2.normalize(hm[...,0], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        heat_disp = cv2.applyColorMap(heat_disp, cv2.COLORMAP_JET)
        heat_disp = cv2.resize(heat_disp, (img_disp.shape[1], img_disp.shape[0]), interpolation=cv2.INTER_NEAREST)

        seg_disp = (seg[...,0] * 255).astype(np.uint8)
        seg_disp = cv2.resize(seg_disp, (img_disp.shape[1], img_disp.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_color = cv2.applyColorMap(seg_disp, cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR), 0.7, seg_color, 0.3, 0)

        stacked = np.hstack([cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR), heat_disp, overlay])

        cv2.imshow("Image | Heatmap | Seg overlay. ESC to quit.", stacked)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
            break
    cv2.destroyAllWindows()

# ============= MAIN =============

def main():
    # parser = argparse.ArgumentParser(description='Board keypoint detector training and evaluation.')
    # parser.add_argument('--annotations', type=str, default='annotations.txt')
    # parser.add_argument('--images', type=str, default='files')
    # parser.add_argument('--epochs', type=int, default=12)
    # parser.add_argument('--batch_size', type=int, default=4)
    # parser.add_argument('--detections', type=int, default=5)
    # args = parser.parse_args()

    img_paths, keypoints = kpu.parse_annotations(ANNOTATIONS_PATH, IMAGES_PATH)

    train_imgs, val_imgs, train_kps, val_kps = train_test_split(
        img_paths, keypoints, test_size=0.3, random_state=42
    )

    train_ds = BoardDataset(train_imgs, train_kps, batch_size=BATCH_SIZE, augmentations=AUGMENTATIONS, num_iterations=5)
    val_ds = BoardDataset(val_imgs, val_kps, batch_size=BATCH_SIZE, augmentations=[identity], num_iterations=0)

    preview_augmented_dataset(train_ds)

    model = build_model()

    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_weights(CHECKPOINT_PATH)
            print(f"Loaded existing weights from {CHECKPOINT_PATH}, continuing training.")
        except Exception as e:
            print(f"Error loading weights, starting from scratch: {e}")
    else:
        print("No checkpoint found, training from scratch.")

    model = train(model, train_ds, val_ds, epochs=EPOCHS)

    model.save(CHECKPOINT_PATH)
    print(f"Model weights saved to {CHECKPOINT_PATH}")

    max_vis = min(DETECTIONS, len(val_imgs))
    for i in range(max_vis):
        print(f"Validation sample {i + 1}/{max_vis}: {val_imgs[i]}")
        infer_and_show(model, val_imgs[i], val_kps[i])
        show_heatmap_comparison(val_imgs[i], val_kps[i], model)

if __name__ == "__main__":
    main()
