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
import matplotlib.pyplot as plt
import matplotlib

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = 128  # Change if needed
BATCH_SIZE = 64
EPOCHS = 500
NETWORK_NAME = 'tile_classifier.pt'  # Model filename
EARLY_STOPPING_PATIENCE = 5  # Stop if val loss doesn't improve for this many epochs

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

def load_tile_images(folder):
    images = []
    tile_labels = []
    crown_labels = []
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            tile, crown = parse_filename(fname)
            if tile in TILE_CLASSES and crown in CROWN_CLASSES:
                img = Image.open(os.path.join(folder, fname)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                images.append(np.array(img) / 255.0)
                tile_labels.append(TILE_CLASSES.index(tile))
                crown_labels.append(CROWN_CLASSES.index(crown))
    return np.array(images), np.array(tile_labels), np.array(crown_labels)

def balance_dataset(images, tile_labels, crown_labels, min_images_per_class=1):
    from collections import defaultdict
    data = defaultdict(list)
    for img, tile, crown in zip(images, tile_labels, crown_labels):
        data[(tile, crown)].append(img)
    # Filter out classes with not enough images (optional)
    filtered_data = {k: v for k, v in data.items() if len(v) >= min_images_per_class}
    if not filtered_data:
        raise ValueError("No classes with enough images. Lower min_images_per_class or add more data.")
    max_count = max(len(imgs) for imgs in filtered_data.values())
    print(f"\nUpsampling all classes to {max_count} images (the largest group).\n")
    balanced_images, tile_labels_out, crown_labels_out = [], [], []
    post_balance_counts = defaultdict(int)
    for (tile, crown), imgs in filtered_data.items():
        sampled_imgs = random.choices(imgs, k=max_count)
        balanced_images.extend(sampled_imgs)
        tile_labels_out.extend([tile] * max_count)
        crown_labels_out.extend([crown] * max_count)
        post_balance_counts[(tile, crown)] = max_count
    print("Image counts per class (tile, crown) AFTER upsampling:")
    for (tile, crown), count in sorted(post_balance_counts.items()):
        print(f"  ({TILE_CLASSES[tile]}, {CROWN_CLASSES[crown]}): {count} images")
    print(f"\nTotal dataset size after upsampling: {len(balanced_images)} images\n")
    return np.array(balanced_images), np.array(tile_labels_out), np.array(crown_labels_out)

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

# -------------------------------
# MODEL DEFINITION
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
    images, tile_labels, crown_labels = load_tile_images('tiles')
    # Split before balancing
    X_train, X_val, y_tile_train, y_tile_val, y_crown_train, y_crown_val = train_test_split(
        images, tile_labels, crown_labels, test_size=0.2, random_state=22
    )
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
    X_train_bal, y_tile_train_bal, y_crown_train_bal = balance_dataset(X_train, y_tile_train, y_crown_train, min_images_per_class=1)
    train_ds = TileDataset(X_train_bal, y_tile_train_bal, y_crown_train_bal, augment=True)
    val_ds = TileDataset(X_val, y_tile_val, y_crown_val, augment=False)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    model = TileNet(len(TILE_CLASSES), len(CROWN_CLASSES)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()

    # Load model if exists
    if os.path.exists(NETWORK_NAME):
        print(f"Loading existing model weights from {NETWORK_NAME}...")
        model.load_state_dict(torch.load(NETWORK_NAME, map_location=device))
        print("Model loaded. Continuing training.")
    else:
        print("No existing model found. Starting training from scratch.")

    # --- Interactive periodic preview of 9 random augmented images with labels (OpenCV) ---
    n_samples = len(train_ds)
    indices = list(range(n_samples))
    batch_size = 9
    stop_preview = False
    while not stop_preview:
        rand_idxs = random.sample(indices, min(batch_size, n_samples))
        imgs = []
        for idx in rand_idxs:
            img, tile_idx, crown_idx = train_ds[idx]
            tile_name = CODE2TERR[TILE_CLASSES[tile_idx]]
            label = f"{tile_name}, {CROWN_CLASSES[crown_idx]}c"
            img_disp = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
            img_disp = draw_label(img_disp, label)
            imgs.append(img_disp)
        # Compose 3x3 grid
        blank = np.ones_like(imgs[0]) * 200
        while len(imgs) < 9:
            imgs.append(blank.astype(np.uint8))
        row1 = np.concatenate(imgs[0:3], axis=1)
        row2 = np.concatenate(imgs[3:6], axis=1)
        row3 = np.concatenate(imgs[6:9], axis=1)
        grid = np.concatenate([row1, row2, row3], axis=0)
        cv2.imshow("9 Random Augmented Images (ENTER=next, ESC=continue to training)", grid)
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
    best_checkpoint_path = 'best_tile_classifier.pt'
    epochs_since_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, train_acc_tile, train_acc_crown = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc_tile, val_acc_crown = run_epoch(model, val_loader, criterion, None, device)

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
              f"Train Acc Crown: {train_acc_crown:.3f} | Val Acc Crown: {val_acc_crown:.3f}")

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

    torch.save(model.state_dict(), NETWORK_NAME)
    print('Training done and model saved!')

    print("Manual verification loop starting. Press ENTER to see next image, ESC to quit.")
    for i in range(len(val_ds)):
        img, tile_label, crown_label = val_ds[i]
        img_display = (img.permute(1,2,0).numpy() * 255).astype(np.uint8)
        img_display = cv2.cvtColor(img_display, cv2.COLOR_RGB2BGR)
        with torch.no_grad():
            model.eval()
            tile_logits, crown_logits = model(img.unsqueeze(0).to(device))
            tile_idx = tile_logits.argmax(1).item()
            crown_idx = crown_logits.argmax(1).item()
        tile_name = CODE2TERR[TILE_CLASSES[tile_idx]]
        crown_val = CROWN_CLASSES[crown_idx]
        label = f"{tile_name}, {crown_val}c"
        img_display = draw_label(img_display, label)
        cv2.imshow("Validation Prediction", img_display)
        key = cv2.waitKey(0)
        if key == 27:
            print("Exiting manual validation viewer.")
            cv2.destroyAllWindows()
            break

if __name__ == "__main__":
    main()
