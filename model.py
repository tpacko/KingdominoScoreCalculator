import torch
import torch.nn as nn
import torch.nn.functional as F


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

if __name__ == "__main__":
    # Sanity Check
    model = BoardKeypointNet()
    dummy_in = torch.randn(2, 3, 512, 512)
    hm, off, seg = model(dummy_in)

    print(f"Input: {dummy_in.shape}")
    print(f"Heatmap: {hm.shape} (Expect 64x64)")
    print(f"Offsets: {off.shape} (Expect 2 channels, 64x64)")
    print(f"SegMask: {seg.shape} (Expect 64x64)")