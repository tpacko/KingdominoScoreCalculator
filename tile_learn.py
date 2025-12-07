import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from PIL import Image
import random
from collections import defaultdict
import cv2
from torchvision import transforms
import torchvision.transforms.functional as TF
import matplotlib.pyplot as plt
import matplotlib

from losses import focal_loss


class FocalLoss(nn.Module):
    """Focal Loss for heatmap regression to handle class imbalance."""
    def __init__(self, alpha=2.0, gamma=4.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: predicted heatmap (B, H, W) or (B, 1, H, W)
            target: target heatmap (B, H, W) or (B, 1, H, W)
        """
        # Ensure both pred and target are (B, H, W)
        if pred.dim() == 4:
            pred = pred.squeeze(1)
        if target.dim() == 4:
            target = target.squeeze(1)

        # Ensure shapes match
        if pred.shape != target.shape:
            raise ValueError(f"Shape mismatch: pred {pred.shape} vs target {target.shape}")

        pos_mask = target.eq(1).float()
        neg_mask = target.lt(1).float()

        pos_loss = torch.log(pred + 1e-12) * torch.pow(1 - pred, self.alpha) * pos_mask
        neg_loss = torch.log(1 - pred + 1e-12) * torch.pow(pred, self.alpha) * torch.pow(1 - target, self.gamma) * neg_mask

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        num_pos = pos_mask.sum()
        if num_pos == 0:
            return -neg_loss
        return -(pos_loss + neg_loss) / num_pos


# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = 128  # Change if needed
HEATMAP_SIZE = 32  # Heatmap output size
GAUSSIAN_SIGMA = 2.0  # Sigma for Gaussian blobs around crown positions
BATCH_SIZE = 64
EPOCHS = 1000
LEARNING_RATE = 1e-5  # Learning rate for optimizer
NETWORK_NAME = 'tile_classifier.pt'  # Model filename (old network)
NETWORK_NAME_KEYPOINT = 'tile_classifier_keypoint.pt'  # New model with heatmap
EARLY_STOPPING_PATIENCE = 10  # Stop if val loss doesn't improve for this many epochs
TILES_FOLDER = 'tiles'  # Path to tiles folder

# Choose heatmap loss function here:
HEATMAP_LOSS = nn.MSELoss()  # Uncomment for MSE loss
# HEATMAP_LOSS = FocalLoss(alpha=2, gamma=4)  # Uncomment for Focal loss

CODE2TERR = {
    "f": "forest",
    "me": "meadow",
    "mi": "mine",
    "w": "water",
    "wa": "wasteland",
    "wh": "wheat",
    "c": "castle"
}

TILE_CLASSES = list(CODE2TERR.keys())
CROWN_CLASSES = [0, 1, 2, 3]

# -------------------------------
# DATA LOADING + AUGMENTATION
# -------------------------------

def parse_filename(filename):
    name = filename.split('.')[0]
    parts = name.split('_')
    tile = ''.join([c for c in parts[0] if not c.isdigit()])
    crown = 0
    if len(parts) > 1:
        for p in parts[1:]:
            if p.startswith('c'):
                crown = int(p[1:])
    return tile, crown

def rotate_points(points, angle, center=(64, 64)):
    """
    Rotate points around center by angle (in degrees) to match PIL/TF.rotate behavior.

    Args:
        points: List of (x, y) tuples
        angle: Rotation angle in degrees (positive = counter-clockwise in PIL)
        center: Center of rotation (x, y)

    Returns:
        List of rotated (x, y) tuples

    Note: PIL uses top-left origin with Y-axis pointing DOWN, so we negate
    the angle to get correct rotation in standard math coordinates.
    """
    import math
    # Negate angle because PIL's Y-axis points downward
    angle_rad = math.radians(-angle)
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)

    rotated = []
    for x, y in points:
        # Translate to origin
        x_rel = x - center[0]
        y_rel = y - center[1]

        # Rotate
        x_rot = x_rel * cos_a - y_rel * sin_a
        y_rot = x_rel * sin_a + y_rel * cos_a

        # Translate back
        x_new = x_rot + center[0]
        y_new = y_rot + center[1]

        rotated.append((x_new, y_new))

    return rotated

def generate_heatmap(crown_positions, original_size, img_size, heatmap_size, sigma):
    """
    Generate a single-channel heatmap with Gaussian blobs at crown positions.

    Args:
        crown_positions: List of (x, y) tuples in original image coordinates
        original_size: Original image size (width, height) tuple
        img_size: Resized image size (128x128)
        heatmap_size: Output heatmap size (32x32)
        sigma: Gaussian sigma in heatmap pixels

    Returns:
        Heatmap as numpy array of shape (heatmap_size, heatmap_size)
    """
    heatmap = np.zeros((heatmap_size, heatmap_size), dtype=np.float32)

    if len(crown_positions) == 0:
        return heatmap

    # Create coordinate grids
    y_grid, x_grid = np.meshgrid(np.arange(heatmap_size), np.arange(heatmap_size), indexing='ij')

    for x_orig, y_orig in crown_positions:
        # Scale from original image to IMG_SIZE
        x_scaled = x_orig * img_size / original_size[0]
        y_scaled = y_orig * img_size / original_size[1]

        # Scale from IMG_SIZE to heatmap_size
        x_hm = x_scaled * heatmap_size / img_size
        y_hm = y_scaled * heatmap_size / img_size

        # Generate Gaussian blob
        gaussian = np.exp(-((x_grid - x_hm)**2 + (y_grid - y_hm)**2) / (2 * sigma**2))
        heatmap = np.maximum(heatmap, gaussian)

    return heatmap


def load_tile_images(folder):
    images = []
    tile_labels = []
    crown_labels = []
    crown_positions = []
    original_sizes = []
    ann_path = os.path.join(folder, 'annotations.txt')
    with open(ann_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 4:
                continue  # skip malformed lines
            fname, tile, crown_count = parts[:3]
            crown_count = int(crown_count)
            crowns_raw = parts[3].split(';') if len(parts) > 3 else []
            # Parse crown positions
            crowns = []
            for i in range(crown_count):
                idx = i * 2
                if idx + 1 < len(crowns_raw):
                    x, y = int(crowns_raw[idx]), int(crowns_raw[idx + 1])
                    if x != -1 and y != -1:
                        crowns.append((x, y))
            # Only load images with valid tile and crown class
            if tile in TILE_CLASSES and crown_count in CROWN_CLASSES:
                img_path = os.path.join(folder, fname)
                if os.path.exists(img_path):
                    img_pil = Image.open(img_path).convert('RGB')
                    orig_size = img_pil.size  # (width, height)
                    img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE))
                    images.append(np.array(img_pil) / 255.0)
                    tile_labels.append(TILE_CLASSES.index(tile))
                    crown_labels.append(CROWN_CLASSES.index(crown_count))
                    crown_positions.append(crowns)
                    original_sizes.append(orig_size)
    return np.array(images), np.array(tile_labels), np.array(crown_labels), crown_positions, original_sizes

def balance_dataset(images, tile_labels, crown_labels, crown_positions, original_sizes, min_images_per_class=1):
    from collections import defaultdict
    data = defaultdict(list)
    for img, tile, crown, pos, orig_size in zip(images, tile_labels, crown_labels, crown_positions, original_sizes):
        data[(tile, crown)].append((img, pos, orig_size))
    # Filter out classes with not enough images (optional)
    filtered_data = {k: v for k, v in data.items() if len(v) >= min_images_per_class}
    if not filtered_data:
        raise ValueError("No classes with enough images. Lower min_images_per_class or add more data.")
    max_count = max(len(imgs) for imgs in filtered_data.values())
    print(f"\nUpsampling all classes to {max_count} images (the largest group).\n")
    balanced_images, tile_labels_out, crown_labels_out, crown_positions_out, original_sizes_out = [], [], [], [], []
    post_balance_counts = defaultdict(int)
    for (tile, crown), items in filtered_data.items():
        sampled_items = random.choices(items, k=max_count)
        for img, pos, orig_size in sampled_items:
            balanced_images.append(img)
            crown_positions_out.append(pos)
            original_sizes_out.append(orig_size)
        tile_labels_out.extend([tile] * max_count)
        crown_labels_out.extend([crown] * max_count)
        post_balance_counts[(tile, crown)] = max_count
    print("Image counts per class (tile, crown) AFTER upsampling:")
    for (tile, crown), count in sorted(post_balance_counts.items()):
        print(f"  ({TILE_CLASSES[tile]}, {CROWN_CLASSES[crown]}): {count} images")
    print(f"\nTotal dataset size after upsampling: {len(balanced_images)} images\n")
    return np.array(balanced_images), np.array(tile_labels_out), np.array(crown_labels_out), crown_positions_out, original_sizes_out

class TileDataset(Dataset):
    def __init__(self, images, tile_labels, crown_labels, augment=False):
        self.images = images
        self.tile_labels = tile_labels
        self.crown_labels = crown_labels
        self.augment = augment
        self.transform = transforms.Compose([
            transforms.ToPILImage(),

            # --- geometric transforms ---
            transforms.RandomApply([
                transforms.RandomChoice([
                    transforms.RandomRotation((0, 0)),
                    transforms.RandomRotation((90, 90)),
                    transforms.RandomRotation((180, 180)),
                    transforms.RandomRotation((270, 270)),
                ])
            ], p=0.75),

            transforms.RandomApply([
                transforms.RandomRotation(10)
            ], p=0.5),

            # --- color transforms ---
            transforms.RandomApply([
                transforms.Grayscale(num_output_channels=3)
            ], p=0.55),  # 15% grayscale

            transforms.RandomApply([
                transforms.ColorJitter(
                    brightness=0.15,
                    hue=0.05,
                    saturation=0.6
                )
            ], p=0.5),  # 50% jitter

            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 0.5))
            ], p=0.3),  # 30% blur

            transforms.ToTensor(),

            # --- noise transforms ---
            transforms.RandomApply([
                transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.02, 0, 1))
            ], p=0.7),  # subtle noise

            transforms.RandomApply([
                transforms.Lambda(lambda x: torch.clamp(x + torch.randn_like(x) * 0.1, 0, 1))
            ], p=0.2),  # stronger noise

            # --- tensor-only ---
            transforms.RandomErasing(
                p=0.25,
                scale=(0.02, 0.1),
                ratio=(0.3, 3.3)
            )
        ])

    def __len__(self):
        return len(self.images)
    def __getitem__(self, idx):
        img = (self.images[idx] * 255).astype(np.uint8)
        if self.augment:
            img = self.transform(img)
        else:
            img = torch.tensor(self.images[idx].transpose(2, 0, 1), dtype=torch.float32)
        tile_label = int(self.tile_labels[idx])
        crown_label = int(self.crown_labels[idx])
        return img, tile_label, crown_label

class TileDatasetWithHeatmap(Dataset):
    """Dataset that returns image, labels, and heatmap. Applies geometric transforms to both."""
    def __init__(self, images, tile_labels, crown_labels, crown_positions, original_sizes, augment=False):
        self.images = images
        self.tile_labels = tile_labels
        self.crown_labels = crown_labels
        self.crown_positions = crown_positions
        self.original_sizes = original_sizes
        self.augment = augment

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = (self.images[idx] * 255).astype(np.uint8)
        crown_pos = self.crown_positions[idx]
        orig_size = self.original_sizes[idx]

        if self.augment:
            # Scale crown positions to IMG_SIZE coordinates first
            crown_pos_scaled = [
                (x * IMG_SIZE / orig_size[0], y * IMG_SIZE / orig_size[1])
                for x, y in crown_pos
            ]

            img_pil = Image.fromarray(img)

            # Track total rotation angle
            total_rotation = 0

            # Random 90-degree rotation
            if random.random() < 0.75:
                angle = random.choice([0, 90, 180, 270])
                if angle != 0:
                    img_pil = TF.rotate(img_pil, angle)
                    total_rotation += angle

            # Random small rotation
            if random.random() < 0.5:
                angle = random.uniform(-10, 10)
                img_pil = TF.rotate(img_pil, angle)
                total_rotation += angle

            # Apply rotation to crown positions
            if total_rotation != 0:
                crown_pos_scaled = rotate_points(crown_pos_scaled, total_rotation, center=(IMG_SIZE/2, IMG_SIZE/2))

            # NOW generate heatmap with rotated coordinates (no interpolation artifacts!)
            heatmap = generate_heatmap(crown_pos_scaled, (IMG_SIZE, IMG_SIZE), IMG_SIZE, HEATMAP_SIZE, GAUSSIAN_SIGMA)

            # Convert to tensors
            img = TF.to_tensor(img_pil)
            heatmap = torch.from_numpy(heatmap).float()

            # Color transforms (only on image, not heatmap)
            if random.random() < 0.55:
                img = TF.rgb_to_grayscale(img, num_output_channels=3)

            if random.random() < 0.5:
                img = TF.adjust_brightness(img, random.uniform(0.85, 1.15))
                img = TF.adjust_saturation(img, random.uniform(0.4, 1.6))
                img = TF.adjust_hue(img, random.uniform(-0.05, 0.05))

            if random.random() < 0.3:
                sigma_val = random.uniform(0.1, 0.5)
                img = TF.gaussian_blur(img, kernel_size=[3, 3], sigma=[sigma_val, sigma_val])

            # Noise (only on image)
            if random.random() < 0.7:
                img = torch.clamp(img + torch.randn_like(img) * 0.02, 0, 1)

            if random.random() < 0.2:
                img = torch.clamp(img + torch.randn_like(img) * 0.1, 0, 1)

            # Random erasing (only on image)
            if random.random() < 0.25:
                i, j, h, w, v = transforms.RandomErasing.get_params(
                    img, scale=(0.02, 0.1), ratio=(0.3, 3.3), value=[0]
                )
                img = TF.erase(img, i, j, h, w, v)
        else:
            # No augmentation - generate heatmap normally
            heatmap = generate_heatmap(crown_pos, orig_size, IMG_SIZE, HEATMAP_SIZE, GAUSSIAN_SIGMA)
            img = torch.tensor(self.images[idx].transpose(2, 0, 1), dtype=torch.float32)
            heatmap = torch.from_numpy(heatmap).float()

        tile_label = int(self.tile_labels[idx])
        crown_label = int(self.crown_labels[idx])
        return img, tile_label, crown_label, heatmap

# -------------------------------
# MAIN FUNCTION
# -------------------------------

class TileNet(nn.Module):
    def __init__(self, num_tile_classes = len(TILE_CLASSES), num_crown_classes = len(CROWN_CLASSES)):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 64x64

            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 32x32

            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.MaxPool2d(2),  # 16x16

            nn.AdaptiveAvgPool2d((4, 4)),  # Flexible pooling
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.tile_head = nn.Linear(512, num_tile_classes)
        self.crown_head = nn.Linear(512, num_crown_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.fc(x)
        tile_logits = self.tile_head(x)
        crown_logits = self.crown_head(x)
        return tile_logits, crown_logits

class TileNetWithHeatmap(nn.Module):
    """TileNet with additional heatmap head for crown localization."""
    def __init__(self, num_tile_classes = len(TILE_CLASSES), num_crown_classes = len(CROWN_CLASSES)):
        super().__init__()
        # Shared feature extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
        )
        self.pool1 = nn.MaxPool2d(2)  # 64x64

        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(),
        )
        self.pool2 = nn.MaxPool2d(2)  # 32x32

        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(),
        )
        self.pool3 = nn.MaxPool2d(2)  # 16x16

        # Classification heads (pooled features)
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.tile_head = nn.Linear(512, num_tile_classes)
        self.crown_head = nn.Linear(512, num_crown_classes)

        # Heatmap head (from 32x32 features)
        # We'll use features from conv2 (which are at 32x32 resolution)
        self.heatmap_head = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1), nn.BatchNorm2d(32), nn.LeakyReLU(),
            nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.LeakyReLU(),
            nn.Conv2d(16, 1, 1),  # 1 channel output
            nn.Sigmoid()  # Output in [0, 1] range
        )

    def forward(self, x):
        # Forward through conv layers
        x = self.conv1(x)
        x = self.pool1(x)  # 64x64

        x = self.conv2(x)
        x = self.pool2(x)  # 32x32
        feat_32x32 = x  # Save 32x32 features for heatmap

        x = self.conv3(x)
        x = self.pool3(x)  # 16x16

        # Classification outputs
        x_cls = self.fc(x)
        tile_logits = self.tile_head(x_cls)
        crown_logits = self.crown_head(x_cls)

        # Heatmap output (using 32x32 features after pool2)
        heatmap = self.heatmap_head(feat_32x32)
        heatmap = heatmap.squeeze(1)  # Remove channel dimension: (B, 1, 32, 32) -> (B, 32, 32)

        return tile_logits, crown_logits, heatmap


# -------------------------------
# MAIN FUNCTION
# -------------------------------

def draw_label(img, text, pos=(5, 20)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color = (0, 0, 0)
    thickness = 2
    img = cv2.putText(img, text, pos, font, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    img = cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def overlay_heatmap(img, heatmap):
    """
    Overlay heatmap on image for visualization.
    Args:
        img: RGB image as numpy array (H, W, 3) in range [0, 255]
        heatmap: Heatmap as numpy array (H, W) in range [0, 1]
    Returns:
        Image with heatmap overlay as numpy array (H, W, 3)
    """
    # Resize heatmap to match image size
    heatmap_resized = cv2.resize(heatmap, (img.shape[1], img.shape[0]))

    # Convert heatmap to color (red)
    heatmap_colored = np.zeros_like(img)
    heatmap_colored[:, :, 2] = (heatmap_resized * 255).astype(np.uint8)  # Red channel

    # Blend with original image
    overlay = cv2.addWeighted(img, 0.7, heatmap_colored, 0.3, 0)
    return overlay


def run_epoch(model, loader, criterion, optimizer=None, device=None):
    if device is None:
        device = torch.device('cpu')
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss, total_acc_tile, total_acc_crown = 0, 0, 0
    total_samples = 0
    for imgs, tile_targets, crown_targets in loader:
        imgs = imgs.to(device)
        tile_targets = torch.as_tensor(tile_targets, dtype=torch.long, device=device)
        crown_targets = torch.as_tensor(crown_targets, dtype=torch.long, device=device)
        if is_train:
            optimizer.zero_grad()
        tile_logits, crown_logits = model(imgs)
        loss_tile = criterion(tile_logits, tile_targets)
        loss_crown = criterion(crown_logits, crown_targets)
        loss = loss_tile + loss_crown
        if is_train:
            loss.backward()
            optimizer.step()
        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_acc_tile += torch.as_tensor(tile_logits.argmax(1).cpu() == tile_targets.cpu()).float().sum().item()
        total_acc_crown += torch.as_tensor(crown_logits.argmax(1).cpu() == crown_targets.cpu()).float().sum().item()
        total_samples += batch_size
    avg_loss = total_loss / total_samples
    avg_acc_tile = total_acc_tile / total_samples
    avg_acc_crown = total_acc_crown / total_samples
    return avg_loss, avg_acc_tile, avg_acc_crown

def run_epoch_with_heatmap(model, loader, criterion_cls, criterion_heatmap, optimizer=None, device=None):
    """Run one epoch with heatmap output."""
    if device is None:
        device = torch.device('cpu')
    is_train = optimizer is not None
    if is_train:
        model.train()
    else:
        model.eval()
    total_loss, total_acc_tile, total_acc_crown = 0, 0, 0
    total_heatmap_loss = 0
    total_samples = 0

    for imgs, tile_targets, crown_targets, heatmap_targets in loader:
        imgs = imgs.to(device)
        tile_targets = torch.as_tensor(tile_targets, dtype=torch.long, device=device)
        crown_targets = torch.as_tensor(crown_targets, dtype=torch.long, device=device)
        heatmap_targets = heatmap_targets.to(device)

        if is_train:
            optimizer.zero_grad()

        tile_logits, crown_logits, heatmap_pred = model(imgs)
        loss_tile = criterion_cls(tile_logits, tile_targets)
        loss_crown = criterion_cls(crown_logits, crown_targets)
        loss_heatmap = criterion_heatmap(heatmap_pred, heatmap_targets)

        # Combine losses (weight heatmap loss appropriately)
        loss = loss_tile + loss_crown + loss_heatmap * 10.0  # Scale heatmap loss

        if is_train:
            loss.backward()
            optimizer.step()

        batch_size = imgs.size(0)
        total_loss += loss.item() * batch_size
        total_heatmap_loss += loss_heatmap.item() * batch_size
        total_acc_tile += torch.as_tensor(tile_logits.argmax(1).cpu() == tile_targets.cpu()).float().sum().item()
        total_acc_crown += torch.as_tensor(crown_logits.argmax(1).cpu() == crown_targets.cpu()).float().sum().item()
        total_samples += batch_size

    avg_loss = total_loss / total_samples
    avg_acc_tile = total_acc_tile / total_samples
    avg_acc_crown = total_acc_crown / total_samples
    avg_heatmap_loss = total_heatmap_loss / total_samples
    return avg_loss, avg_acc_tile, avg_acc_crown, avg_heatmap_loss


def plot_training_progress(axs, train_losses, val_losses, train_acc_tile_hist, val_acc_tile_hist, train_acc_crown_hist, val_acc_crown_hist):
    axs[0, 0].clear()
    axs[0, 0].plot(train_losses, label='Train Loss')
    axs[0, 0].plot(val_losses, label='Val Loss')
    axs[0, 0].set_title('Total Loss')
    axs[0, 0].set_xlabel('Epoch')
    axs[0, 0].set_ylabel('Loss')
    axs[0, 0].legend()
    axs[0, 0].grid(True)

    axs[0, 1].clear()
    axs[0, 1].plot(train_acc_tile_hist, label='Train Acc Tile')
    axs[0, 1].plot(val_acc_tile_hist, label='Val Acc Tile')
    axs[0, 1].set_title('Tile Accuracy')
    axs[0, 1].set_xlabel('Epoch')
    axs[0, 1].set_ylabel('Accuracy')
    axs[0, 1].legend()
    axs[0, 1].grid(True)

    axs[1, 0].clear()
    axs[1, 0].plot(train_acc_crown_hist, label='Train Acc Crown')
    axs[1, 0].plot(val_acc_crown_hist, label='Val Acc Crown')
    axs[1, 0].set_title('Crown Accuracy')
    axs[1, 0].set_xlabel('Epoch')
    axs[1, 0].set_ylabel('Accuracy')
    axs[1, 0].legend()
    axs[1, 0].grid(True)

    axs[1, 1].axis('off')
    plt.tight_layout()
    plt.pause(0.1)

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    images, tile_labels, crown_labels, crown_positions, original_sizes = load_tile_images(TILES_FOLDER)

    # Split before balancing - need to split all arrays including crown_positions and original_sizes
    indices = np.arange(len(images))
    train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=22)

    X_train = images[train_indices]
    X_val = images[val_indices]
    y_tile_train = tile_labels[train_indices]
    y_tile_val = tile_labels[val_indices]
    y_crown_train = crown_labels[train_indices]
    y_crown_val = crown_labels[val_indices]
    crown_pos_train = [crown_positions[i] for i in train_indices]
    crown_pos_val = [crown_positions[i] for i in val_indices]
    orig_sizes_train = [original_sizes[i] for i in train_indices]
    orig_sizes_val = [original_sizes[i] for i in val_indices]
    # Print number of loaded classes for train and validation set
    def print_class_counts(tile_labels, crown_labels, set_name):
        from collections import Counter
        counts = Counter(zip(tile_labels, crown_labels))
        print(f"\nLoaded class counts for {set_name} set:")
        for (tile, crown), count in sorted(counts.items()):
            print(f"  ({TILE_CLASSES[tile]}, {CROWN_CLASSES[crown]}): {count} images")
        print(f"Total: {sum(counts.values())} images in {set_name} set\n")

    print_class_counts(y_tile_train, y_crown_train, 'train')
    print_class_counts(y_tile_val, y_crown_val, 'validation')

    # Balance only the training set
    X_train_bal, y_tile_train_bal, y_crown_train_bal, crown_pos_train_bal, orig_sizes_train_bal = balance_dataset(
        X_train, y_tile_train, y_crown_train, crown_pos_train, orig_sizes_train, min_images_per_class=1
    )

    # Use new heatmap-enabled dataset
    train_ds = TileDatasetWithHeatmap(X_train_bal, y_tile_train_bal, y_crown_train_bal, crown_pos_train_bal, orig_sizes_train_bal, augment=True)
    val_ds = TileDatasetWithHeatmap(X_val, y_tile_val, y_crown_val, crown_pos_val, orig_sizes_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Use new heatmap-enabled model
    model = TileNetWithHeatmap(len(TILE_CLASSES), len(CROWN_CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion_cls = nn.CrossEntropyLoss()
    criterion_heatmap = HEATMAP_LOSS  # Use config variable for heatmap loss

    # Load model if exists
    if os.path.exists(NETWORK_NAME_KEYPOINT):
        print(f"Loading existing model weights from {NETWORK_NAME_KEYPOINT}...")
        model.load_state_dict(torch.load(NETWORK_NAME_KEYPOINT, map_location=device))
        print("Model loaded. Continuing training.")
    else:
        print("No existing model found. Starting training from scratch.")

    # --- Interactive periodic preview of 9 random augmented images with heatmaps (OpenCV) ---
    n_samples = len(train_ds)
    indices = list(range(n_samples))
    batch_size = 9
    stop_preview = False
    while not stop_preview:
        rand_idxs = random.sample(indices, min(batch_size, n_samples))
        imgs = []
        heatmaps = []
        for idx in rand_idxs:
            img, tile_idx, crown_idx, heatmap = train_ds[idx]
            tile_name = CODE2TERR[TILE_CLASSES[tile_idx]]
            label = f"{tile_name}, {CROWN_CLASSES[crown_idx]}c"
            img_disp = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
            img_disp = draw_label(img_disp, label)
            imgs.append(img_disp)

            # Visualize heatmap as grayscale (resize to match image size)
            heatmap_np = heatmap.numpy()
            heatmap_resized = cv2.resize(heatmap_np, (img_disp.shape[1], img_disp.shape[0]))
            heatmap_gray = (heatmap_resized * 255).astype(np.uint8)
            heatmap_vis = cv2.cvtColor(heatmap_gray, cv2.COLOR_GRAY2BGR)  # Convert to 3-channel for display
            heatmaps.append(heatmap_vis)

        # Compose 3x3 grids for both images and heatmaps
        blank = np.ones_like(imgs[0]) * 200
        while len(imgs) < 9:
            imgs.append(blank.astype(np.uint8))
            heatmaps.append(blank.astype(np.uint8))

        # Create image grid
        row1_img = np.concatenate(imgs[0:3], axis=1)
        row2_img = np.concatenate(imgs[3:6], axis=1)
        row3_img = np.concatenate(imgs[6:9], axis=1)
        grid_img = np.concatenate([row1_img, row2_img, row3_img], axis=0)

        # Create heatmap grid (grayscale)
        row1_hm = np.concatenate(heatmaps[0:3], axis=1)
        row2_hm = np.concatenate(heatmaps[3:6], axis=1)
        row3_hm = np.concatenate(heatmaps[6:9], axis=1)
        grid_hm = np.concatenate([row1_hm, row2_hm, row3_hm], axis=0)

        # Stack vertically: images on top, grayscale heatmaps on bottom
        grid_combined = np.concatenate([grid_img, grid_hm], axis=0)

        cv2.imshow("Augmented Images (top) + Grayscale Heatmaps (bottom) | ENTER=next, ESC=continue", grid_combined)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            stop_preview = True
            cv2.destroyAllWindows()
        # Otherwise (ENTER, etc.), just continue the loop

    print("Continuing to learning...")

    # --------- Model Training ---------
    matplotlib.use('TkAgg')
    plt.ion()
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    train_losses, val_losses = [], []
    train_acc_tile_hist, val_acc_tile_hist = [], []
    train_acc_crown_hist, val_acc_crown_hist = [], []
    best_val_loss = float('inf')
    best_checkpoint_path = 'best_tile_classifier_keypoint.pt'
    epochs_since_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc_tile, train_acc_crown, train_hm_loss = run_epoch_with_heatmap(
            model, train_loader, criterion_cls, criterion_heatmap, optimizer, device
        )
        val_loss, val_acc_tile, val_acc_crown, val_hm_loss = run_epoch_with_heatmap(
            model, val_loader, criterion_cls, criterion_heatmap, None, device
        )

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_acc_tile_hist.append(train_acc_tile)
        val_acc_tile_hist.append(val_acc_tile)
        train_acc_crown_hist.append(train_acc_crown)
        val_acc_crown_hist.append(val_acc_crown)

        plot_training_progress(
            axs, train_losses, val_losses,
            train_acc_tile_hist, val_acc_tile_hist,
            train_acc_crown_hist, val_acc_crown_hist
        )

        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
              f"Train Acc Tile: {train_acc_tile:.3f} | Val Acc Tile: {val_acc_tile:.3f} | "
              f"Train Acc Crown: {train_acc_crown:.3f} | Val Acc Crown: {val_acc_crown:.3f} | "
              f"Train HM Loss: {train_hm_loss:.4f} | Val HM Loss: {val_hm_loss:.4f}")

        # Save checkpoint if best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), best_checkpoint_path)
            print(f"Best checkpoint saved at epoch {epoch+1} with val loss {val_loss:.4f}")
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        # Early stopping check
        if epochs_since_improvement >= EARLY_STOPPING_PATIENCE:
            print(f"Early stopping: Validation loss did not improve for {EARLY_STOPPING_PATIENCE} epochs.")
            break

    plt.ioff()
    plt.show()

    torch.save(model.state_dict(), NETWORK_NAME_KEYPOINT)
    print('Training done and model saved!')

    print("Manual verification loop starting. Press ENTER to see next image, ESC to quit.")
    for i in range(len(val_ds)):
        img, tile_label, crown_label, heatmap_gt = val_ds[i]
        img_display = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            model.eval()
            tile_logits, crown_logits, heatmap_pred = model(img.unsqueeze(0).to(device))
            tile_idx = tile_logits.argmax(1).item()
            crown_idx = crown_logits.argmax(1).item()
            heatmap_pred_np = heatmap_pred[0].cpu().numpy()
        tile_name = CODE2TERR[TILE_CLASSES[tile_idx]]
        crown_val = CROWN_CLASSES[crown_idx]
        label = f"{tile_name}, {crown_val}c"
        img_display = draw_label(img_display, label)

        # Show image and heatmap side by side
        # Resize heatmap to match image size and convert to color
        heatmap_resized = cv2.resize(heatmap_pred_np, (img_display.shape[1], img_display.shape[0]))
        heatmap_colored = cv2.applyColorMap((heatmap_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)

        # Concatenate horizontally: image | heatmap
        side_by_side = np.concatenate([img_display, heatmap_colored], axis=1)

        cv2.imshow("Validation: Image | Heatmap (ENTER=next, ESC=quit)", side_by_side)
        key = cv2.waitKey(0)
        if key == 27:
            print("Exiting manual validation viewer.")
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
