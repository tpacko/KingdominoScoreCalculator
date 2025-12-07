import torch
import torch.nn as nn
import torch.nn.functional as F


# =====================================
# BasicBlock: Residual block for deep networks
# -------------------------------------
# Purpose: Used in ResNet-style architectures for board/keypoint detection.
# Input:  image tensor (B, in_planes, H, W)
# Output: tensor (B, planes, H, W)
# Use: Internal block for BoardKeypointNet.
class BasicBlock(nn.Module):
    """
    Standard Residual Block: preserves gradient flow for deep training.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


# =====================================
# BoardKeypointNet: Board keypoint and segmentation detection
# ----------------------------------------------------------
# Purpose: Detects board corners/keypoints, segmentation mask, and offset maps.
# Input:  image tensor (B, 3, 512, 512)
# Output: tuple (heatmap, offsets, segmentation)
#   - heatmap: (B, 1, 64, 64) - probability map for keypoints
#   - offsets: (B, 2, 64, 64) - x/y offset for keypoints
#   - segmentation: (B, 1, 64, 64) - board mask
# Use: Board detection and localization tasks.
class BoardKeypointNet(nn.Module):
    def __init__(self, num_classes=1):
        super(BoardKeypointNet, self).__init__()
        self.in_planes = 64

        # --- ENCODER (ResNet-18 style) ---

        # Stem: Input 512x512 -> 128x128
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Layers
        # Layer 1: 128x128 -> 128x128
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        # Layer 2: 128x128 -> 64x64 (Target Resolution - Save for Skip)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        # Layer 3: 64x64 -> 32x32 (Save for Skip)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        # Layer 4: 32x32 -> 16x16 (Bottleneck)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # --- DECODER (FPN style) ---

        # Upsample 1: 16x16 -> 32x32
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input = 512 (from layer4) + 256 (from layer3)
        self.dec1 = nn.Sequential(
            nn.Conv2d(512 + 256, 256, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Upsample 2: 32x32 -> 64x64
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        # Input = 256 (from dec1) + 128 (from layer2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256 + 128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # --- HEADS ---
        # All heads output at 64x64 resolution

        # Heatmap: Probability of a corner existing
        self.head_hm = nn.Conv2d(128, num_classes, kernel_size=1)

        # Offsets: x, y adjustment (delta) to recover precision lost by downsampling
        self.head_off = nn.Conv2d(128, 2, kernel_size=1)

        # Segmentation: Binary mask of the board
        self.head_seg = nn.Conv2d(128, 1, kernel_size=1)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        # --- Encode ---
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)  # [B, 64, 128, 128]

        c1 = self.layer1(x)  # [B, 64, 128, 128]
        c2 = self.layer2(c1)  # [B, 128, 64, 64]  <-- Skip Target
        c3 = self.layer3(c2)  # [B, 256, 32, 32]  <-- Skip Target
        c4 = self.layer4(c3)  # [B, 512, 16, 16]  <-- Bottleneck

        # --- Decode ---

        # Up 1 (16 -> 32)
        u1 = self.up1(c4)
        u1 = torch.cat((u1, c3), dim=1)  # Skip connection
        d1 = self.dec1(u1)  # [B, 256, 32, 32]

        # Up 2 (32 -> 64)
        u2 = self.up2(d1)
        u2 = torch.cat((u2, c2), dim=1)  # Skip connection
        d2 = self.dec2(u2)  # [B, 128, 64, 64]

        # --- Heads ---
        hm = torch.sigmoid(self.head_hm(d2))

        # NO ACTIVATION on offsets (predicts absolute float values)
        off = self.head_off(d2)

        seg = torch.sigmoid(self.head_seg(d2))

        return hm, off, seg


# =====================================
# KeypointNet: General keypoint and segmentation detection
# -------------------------------------------------------
# Purpose: Detects keypoints, offsets, and segmentation mask for generic images.
# Input:  image tensor (B, 3, H, W)
# Output: tuple (heatmap, offsets, segmentation)
#   - heatmap: (B, 1, H', W') - probability map for keypoints
#   - offsets: (B, 2, H', W') - x/y offset for keypoints
#   - segmentation: (B, 1, H', W') - mask
# Use: Keypoint detection and segmentation tasks.
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

# =====================================
# TileNet: Tile and Crown Classification Model
# -------------------------------------
# Purpose: Classifies tile type and number of crowns from a tile image.
# Input:  image tensor of shape (B, 3, 128, 128)
# Output: tuple (tile_logits, crown_logits)
#   - tile_logits: (B, num_tile_classes)
#   - crown_logits: (B, num_crown_classes)
# Use: For basic tile and crown detection/classification.
class TileNet(nn.Module):
    def __init__(self, num_tile_classes, num_crown_classes):
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

# =====================================
# TileNetWithHeatmap: Tile/Crown Classification + Crown Location Heatmap
# ----------------------------------------------------------------------
# Purpose: Classifies tile type, number of crowns, and predicts a heatmap
#          for crown locations (single-channel, 32x32 output).
# Input:  image tensor of shape (B, 3, 128, 128)
# Output: tuple (tile_logits, crown_logits, heatmap)
#   - tile_logits: (B, num_tile_classes)
#   - crown_logits: (B, num_crown_classes)
#   - heatmap: (B, 32, 32) (single-channel, Gaussian blobs at crown positions)
# Use: For tile/crown classification and crown keypoint localization.
class TileNetWithHeatmap(nn.Module):
    def __init__(self, num_tile_classes, num_crown_classes):
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

# =====================================

if __name__ == "__main__":
    # Sanity Check
    model = BoardKeypointNet()
    dummy_in = torch.randn(2, 3, 512, 512)
    hm, off, seg = model(dummy_in)

    print(f"Input: {dummy_in.shape}")
    print(f"Heatmap: {hm.shape} (Expect 64x64)")
    print(f"Offsets: {off.shape} (Expect 2 channels, 64x64)")
    print(f"SegMask: {seg.shape} (Expect 64x64)")
