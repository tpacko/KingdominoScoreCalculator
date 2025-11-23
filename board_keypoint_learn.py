import os
from time import sleep
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import cv2
import argparse
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import keypoint_utils as kpu
from keypoint_augmentations import (
    identity, rotate_90, rotate_180, rotate_neg90,
    horizontal_flip, vertical_flip, random_small_rotation,
    random_scale, random_translate, color_jitter, random_perspective
)

from model import BoardKeypointNet

import matplotlib.pyplot as plt
import matplotlib

# ============= CONFIGURATION =============

CHECKPOINT_PATH = "board_keypoint_detector.pt"
IMG_SIZE = kpu.IMG_SIZE
HM_SIZE = kpu.HM_SIZE
NUM_CORNERS = kpu.NUM_CORNERS

ANNOTATIONS_PATH = 'annotations.txt'
IMAGES_PATH = 'files'
EPOCHS = 250
BATCH_SIZE = 16
DETECTIONS = 5
LEARNING_RATE = 1e-5

AUGMENTATIONS = [
    identity,
    color_jitter,
    rotate_90, rotate_180, rotate_neg90, horizontal_flip, vertical_flip,
    random_small_rotation, random_scale, random_translate, random_perspective
]


# ============= END CONFIGURATION =============

# ============= Dataset =============

class BoardDataset(Dataset):
    def __init__(self, img_paths, kps, augmentations=None, apply_augmentation=True):
        self.img_paths = img_paths
        self.kps = kps
        self.augmentations = augmentations if augmentations is not None else [identity]
        self.apply_augmentation = apply_augmentation

        # Cache for loaded images
        self._img_cache = {}

        print(f"Dataset initialized with {len(self.img_paths)} samples")

    def _load_and_cache_image(self, idx):
        """Load image once and cache in memory"""
        if idx not in self._img_cache:
            img = cv2.imread(self.img_paths[idx])
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, self.kps[idx], IMG_SIZE)
            self._img_cache[idx] = (img_resized, kps_rescaled)
        return self._img_cache[idx]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        # Get cached image (loads once, reused every epoch)
        img_resized, kps_rescaled = self._load_and_cache_image(idx)

        # Apply augmentation on copies
        if self.apply_augmentation and len(self.augmentations) > 1:
            # random int between 0 and 4
            iterations = random.randint(0, 4)
            for _ in range(iterations):
                 aug_fn = random.choice(self.augmentations)
                 img_resized, kps_rescaled = aug_fn(img_resized.copy(), kps_rescaled.copy())

        # Normalize
        img_resized = img_resized / 255.0

        # Generate targets and convert to tensors
        hm, off, mask, seg = kpu.generate_targets(kps_rescaled)

        img_tensor = torch.tensor(img_resized, dtype=torch.float32).permute(2, 0, 1)
        hm_tensor = torch.tensor(hm, dtype=torch.float32).permute(2, 0, 1)
        off_tensor = torch.tensor(off, dtype=torch.float32).permute(2, 0, 1)
        mask_tensor = torch.tensor(mask, dtype=torch.float32).permute(2, 0, 1)
        seg_tensor = torch.tensor(seg, dtype=torch.float32).permute(2, 0, 1)

        return img_tensor, {
            'heatmap': hm_tensor,
            'offsets': off_tensor,
            'mask': mask_tensor,
            'segmask': seg_tensor
        }

# ============= Model =============

class KeypointNet(nn.Module):
    def __init__(self, base_filters=64):
        super().__init__()
        self.conv1 = nn.Conv2d(3, base_filters, 5, stride=1, padding=2)

        self.conv2 = nn.Conv2d(base_filters, base_filters * 2, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(base_filters * 2, base_filters * 2, 3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(base_filters * 2, base_filters * 2, 3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(base_filters * 2, base_filters * 2, 3, stride=2, padding=1)

        self.conv6 = nn.Conv2d(base_filters * 2, base_filters * 4, 3, stride=1, padding=1)
        self.conv7 = nn.Conv2d(base_filters * 4, base_filters * 4, 3, stride=1, padding=1)
        self.conv8 = nn.Conv2d(base_filters * 4, base_filters * 4, 3, stride=2, padding=1)

        self.conv9 = nn.Conv2d(base_filters * 4, base_filters * 8, 3, stride=1, padding=1)
        self.conv10 = nn.Conv2d(base_filters * 8, base_filters * 8, 3, stride=1, padding=1)
        self.conv11 = nn.Conv2d(base_filters * 8, base_filters * 8, 3, stride=1, padding=1)

        self.heatmap = nn.Conv2d(base_filters * 8, 1, 1)
        self.offsets = nn.Conv2d(base_filters * 8, 2, 1)
        self.segmentation = nn.Conv2d(base_filters * 8, 1, 1)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.relu(self.conv5(x))
        x = self.relu(self.conv6(x))
        x = self.relu(self.conv7(x))
        x = self.relu(self.conv8(x))
        x = self.relu(self.conv9(x))
        x = self.relu(self.conv10(x))
        x = self.relu(self.conv11(x))
        heatmap = self.sigmoid(self.heatmap(x))
        offsets = self.tanh(self.offsets(x))
        segmentation = self.sigmoid(self.segmentation(x))
        return heatmap, offsets, segmentation


# ============= Losses =============

def mse_loss(y_true, y_pred):
    return nn.functional.mse_loss(y_pred, y_true)

def bce_loss(y_true, y_pred):
    return nn.functional.binary_cross_entropy(y_pred, y_true)


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    union = torch.sum(y_true_f) + torch.sum(y_pred_f)
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice


def offset_loss(y_true, y_pred, mask):
    diff = (y_true - y_pred) * mask
    return torch.sum(torch.abs(diff)) / (torch.sum(mask) + 1e-6)


def seg_loss(y_true, y_pred, w_bce=1.0, w_dice=1.0):
    return w_bce * bce_loss(y_true, y_pred) + w_dice * dice_loss(y_true, y_pred)

def focal_loss(pred, gt, alpha=2.0, beta=4.0):
    # pred should be passed through sigmoid already
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, beta)

    loss = 0
    pos_loss = torch.log(pred + 1e-9) * torch.pow(1 - pred, alpha) * pos_inds
    neg_loss = torch.log(1 - pred + 1e-9) * torch.pow(pred, alpha) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss


def focal_mse_loss(pred, gt):
    pos_inds = gt.eq(1).float()
    neg_inds = gt.lt(1).float()

    neg_weights = torch.pow(1 - gt, 4)  # Beta=4

    loss = 0
    pred = torch.clamp(pred, 1e-6, 1 - 1e-6)  # Numerical stability

    # Positive case (center of blob)
    pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds

    # Negative case (background + surrounding gaussian values)
    neg_loss = torch.log(1 - pred) * torch.pow(pred, 2) * neg_weights * neg_inds

    num_pos = pos_inds.float().sum()
    pos_loss = pos_loss.sum()
    neg_loss = neg_loss.sum()

    if num_pos == 0:
        loss = -neg_loss
    else:
        loss = -(pos_loss + neg_loss) / num_pos
    return loss

# ============= Training =============

def train(model, train_loader, val_loader, epochs=15, checkpoint_dir="./checkpoints"):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")

    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    os.makedirs(checkpoint_dir, exist_ok=True)

    best_loss = float("inf")
    train_losses = []
    val_losses = []

    plt.ion()
    fig, ax = plt.subplots(figsize=(8, 5))

    for epoch in range(epochs):
        model.train()
        hm_losses, off_losses, seg_losses = [], [], []
        print(f"\nEpoch {epoch + 1}/{epochs}")

        def loss_fn(output: Any, target: Any):
            # l1 = 1 * focal_loss(hms_true, hms_pred, 0.25, 1.0)
            l1 = 200 * mse_loss(hms_true, hms_pred)
            l2 = 1.0 * offset_loss(offs_true, offs_pred, mask)
            l3 = 1.0 * seg_loss(seg_true, seg_pred, w_bce=1.0, w_dice=1.0)
            loss = l1 + l2 + l3
            return loss, l1, l2, l3

        for imgs, targets in tqdm(train_loader, desc='Training', leave=True):
            imgs = imgs.to(device)
            hms_true = targets['heatmap'].to(device)
            offs_true = targets['offsets'].to(device)
            mask = targets['mask'].to(device)
            seg_true = targets['segmask'].to(device)

            optimizer.zero_grad()
            hms_pred, offs_pred, seg_pred = model(imgs)

            loss, l1, l2, l3 =  loss_fn((hms_pred, offs_pred, seg_pred), (hms_true, offs_true, seg_true))

            loss.backward()
            optimizer.step()

            hm_losses.append(float(l1.item()))
            off_losses.append(float(l2.item()))
            seg_losses.append(float(l3.item()))

        mean_hm_loss = np.mean(hm_losses)
        mean_off_loss = np.mean(off_losses)
        mean_seg_loss = np.mean(seg_losses)
        mean_loss = mean_hm_loss + mean_off_loss + mean_seg_loss
        train_losses.append(mean_loss)

        # Validation
        model.eval()
        val_hm_losses, val_off_losses, val_seg_losses = [], [], []
        with torch.no_grad():
            for imgs, targets in tqdm(val_loader, desc='Validation', leave=True):
                imgs = imgs.to(device)
                hms_true = targets['heatmap'].to(device)
                offs_true = targets['offsets'].to(device)
                mask = targets['mask'].to(device)
                seg_true = targets['segmask'].to(device)

                hms_pred, offs_pred, seg_pred = model(imgs)

                loss, l1, l2, l3 = loss_fn((hms_pred, offs_pred, seg_pred), (hms_true, offs_true, seg_true))

                val_hm_losses.append(float(l1.item()))
                val_off_losses.append(float(l2.item()))
                val_seg_losses.append(float(l3.item()))

        mean_val_hm_loss = np.mean(val_hm_losses)
        mean_val_off_loss = np.mean(val_off_losses)
        mean_val_seg_loss = np.mean(val_seg_losses)
        mean_val_loss = mean_val_hm_loss + mean_val_off_loss + mean_val_seg_loss
        val_losses.append(mean_val_loss)

        # Plot
        ax.clear()
        ax.set_ylim(0, 3)
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

        print(
            f"Train: hm_loss={mean_hm_loss:.4f}, off_loss={mean_off_loss:.4f}, seg_loss={mean_seg_loss:.4f}, total_loss={mean_loss:.4f} | "
            f"Val: hm_loss={mean_val_hm_loss:.4f}, off_loss={mean_val_off_loss:.4f}, seg_loss={mean_val_seg_loss:.4f}, total_loss={mean_val_loss:.4f}")

        if mean_val_loss < best_loss:
            best_loss = mean_val_loss
            print(f"New best validation loss: {best_loss:.4f}, saving model weights...")
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best.pt"))

    plt.ioff()
    plt.show()
    return model


# ============= Evaluation =============

def infer_and_show(model, img_path, kp_true):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, kp_true, IMG_SIZE)

    inp = img_resized / 255.0
    inp = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        hm_pred, off_pred, seg_pred = model(inp)

    hm_pred = hm_pred[0, 0].cpu().numpy()  # [H, W]
    off_pred = off_pred[0].cpu().numpy()  # [2, H, W]
    seg_pred = seg_pred[0, 0].cpu().numpy()  # [H, W]

    # Transpose offsets to [H, W, 2] for reconstruct_keypoints
    off_pred = np.transpose(off_pred, (1, 2, 0))
    hm_exp = np.expand_dims(hm_pred, -1)
    rec_kps = kpu.reconstruct_keypoints(hm_exp, off_pred, threshold=0.5)

    img_show = img_resized.copy()
    if len(rec_kps) > NUM_CORNERS:
        flat = hm_pred.flatten()
        idxs = np.argpartition(-flat, NUM_CORNERS)[:NUM_CORNERS]
        ys, xs = np.unravel_index(idxs, hm_pred.shape)
        rec_kps = [((x + off_pred[y, x, 0]) / (HM_SIZE / IMG_SIZE), (y + off_pred[y, x, 1]) / (HM_SIZE / IMG_SIZE)) for
                   y, x in zip(ys, xs)]

    for rx, ry in rec_kps:
        cv2.circle(img_show, (int(rx), int(ry)), 8, (0, 255, 0), 2)
    for kp in kps_rescaled:
        cv2.drawMarker(img_show, (int(kp[0]), int(kp[1])), (255, 0, 0), markerType=cv2.MARKER_CROSS, thickness=2)

    seg_up = cv2.resize(seg_pred, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)
    seg_color = cv2.applyColorMap((seg_up * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_show, 0.7, seg_color, 0.3, 0)

    plt.imshow(overlay)
    plt.title("Green: Predicted, Blue: GT (rescaled) | Seg overlay")
    plt.show()


def show_heatmap_comparison(img_path, keypoints, model):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized, kps_rescaled = kpu.resize_img_and_kps(img_rgb, keypoints, IMG_SIZE)



    inp = img_resized / 255.0
    inp = torch.tensor(inp, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        hm_pred, off_pred, seg_pred = model(inp)

    hm_pred = hm_pred[0, 0].cpu().numpy()  # [H, W]
    off_pred = off_pred[0].cpu().numpy()  # [2, H, W]
    seg_pred = seg_pred[0, 0].cpu().numpy()  # [H, W]
    off_pred = np.transpose(off_pred, (1, 2, 0))

    hm_gt, _, _, seg_gt = kpu.generate_targets(kps_rescaled)
    hm_gt = hm_gt[..., 0]
    seg_gt = seg_gt[..., 0]

    fig, axes = plt.subplots(1, 5, figsize=(30, 6))
    axes[0].imshow(img_resized);
    axes[0].set_title("Input");
    axes[0].axis('off')
    axes[1].imshow(hm_gt, cmap='hot');
    axes[1].set_title("GT Heatmap");
    axes[1].axis('off')
    axes[2].imshow(hm_pred, cmap='hot');
    axes[2].set_title("Pred Heatmap");
    axes[2].axis('off')
    axes[3].imshow(seg_gt, cmap='gray');
    axes[3].set_title("GT Seg");
    axes[3].axis('off')
    axes[4].imshow(seg_pred, cmap='jet');
    axes[4].set_title("Pred Seg");
    axes[4].axis('off')
    plt.tight_layout()
    plt.show()


# ============= MAIN =============

def main():
    print("PyTorch version:", torch.__version__)
    print("GPUs:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU detected")
    matplotlib.use('TkAgg')

    # Load data
    img_paths, keypoints = kpu.parse_annotations(ANNOTATIONS_PATH, IMAGES_PATH)
    print(f"Loaded {len(img_paths)} images with annotations")

    # Train/val split
    train_imgs, val_imgs, train_kps, val_kps = train_test_split(
        img_paths, keypoints, test_size=0.3, random_state=42
    )
    print(f"Train samples: {len(train_imgs)}, Val samples: {len(val_imgs)}")

    # Create datasets
    train_ds = BoardDataset(train_imgs, train_kps, augmentations=AUGMENTATIONS, apply_augmentation=True)
    val_ds = BoardDataset(val_imgs, val_kps, augmentations=[identity], apply_augmentation=False)

    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    # Create model
    # model = KeypointNet()
    model = BoardKeypointNet()

    # Load existing weights if available
    if os.path.exists(CHECKPOINT_PATH):
        try:
            model.load_state_dict(torch.load(CHECKPOINT_PATH))
            print(f"Loaded existing weights from {CHECKPOINT_PATH}, continuing training.")
        except Exception as e:
            print(f"Error loading weights, starting from scratch: {e}")
    else:
        print("No checkpoint found, training from scratch.")

    # Train
    model = train(model, train_loader, val_loader, epochs=EPOCHS)

    # Save final model
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"Model weights saved to {CHECKPOINT_PATH}")

    # load best weights
    # DEBUG

    # best_weights_path = os.path.join("./checkpoints", "best.pt")
    # if os.path.exists(best_weights_path):
    #     model.load_state_dict(torch.load(best_weights_path))
    #     print(f"Loaded best model weights from {best_weights_path} for evaluation.")
    #
    # show_heatmap_comparison(val_imgs[4], val_kps[4], model)

    # Visualize results on validation set
    max_vis = min(DETECTIONS, len(val_imgs))
    for i in range(max_vis):
        print(f"Validation sample {i + 1}/{max_vis}: {val_imgs[i]}")
        infer_and_show(model, val_imgs[i], val_kps[i])
        show_heatmap_comparison(val_imgs[i], val_kps[i], model)


if __name__ == "__main__":
    main()