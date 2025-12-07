import torch
import torch.nn as nn


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
