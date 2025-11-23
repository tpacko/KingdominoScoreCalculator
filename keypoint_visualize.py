import os
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.ndimage import label

import keypoint_utils
from board_keypoint_learn import KeypointNet
from model import BoardKeypointNet

# ==== USER-EDITABLE PARAMETERS (model + folders only!) ====
# MODEL_PATH      = "board_keypoint_detector.h5"
MODEL_PATH      = "checkpoints/best.pt"
IMG_FOLDER      = "files"
HEATMAP_THRESH  = 0.85   # Minimum probability for blob detection
SHOW_MAX_IMAGES = 50
# ================================================

def resize_img(img, desired_size):
    h0, w0 = img.shape[:2]
    img_resized = cv2.resize(img, (desired_size, desired_size))
    return img_resized

def blobs_from_heatmap(hm, min_prob=0.95):
    mask = (hm >= min_prob).astype(np.uint8)
    labeled, num = label(mask)
    return labeled, num

def get_blob_peaks(labeled, num, hm):
    blob_props = []
    for i in range(1, num+1):
        ys, xs = np.where(labeled == i)
        if len(xs) == 0:
            continue
        max_idx = np.argmax(hm[ys, xs])
        peak_y, peak_x = ys[max_idx], xs[max_idx]
        blob_area = len(xs)
        blob_props.append({'area': blob_area, 'peak_y': peak_y, 'peak_x': peak_x, 'prob': hm[peak_y, peak_x]})
    # Sort by area descending
    blob_props.sort(key=lambda x: -x['area'])
    return blob_props

def predict_corners(model, img_resized):
    inp = torch.from_numpy(img_resized).float().permute(2, 0, 1).unsqueeze(0) / 255.0
    with torch.no_grad():
        outputs = model(inp)
    hm_pred, off_pred, seg_pred = outputs
    hm_pred = hm_pred.squeeze().cpu().numpy()
    off_pred = off_pred.squeeze().cpu().numpy()
    seg_pred = seg_pred.squeeze().cpu().numpy()

    labeled, num = blobs_from_heatmap(hm_pred, min_prob=HEATMAP_THRESH)
    blobs = get_blob_peaks(labeled, num, hm_pred)

    corners = []
    for blob in blobs[:keypoint_utils.NUM_CORNERS]:
        x, y = blob['peak_x'], blob['peak_y']
        ox, oy = off_pred[:, y, x]
        fx = (x + ox) * keypoint_utils.IMG_SIZE / keypoint_utils.HM_SIZE
        fy = (y + oy) * keypoint_utils.IMG_SIZE / keypoint_utils.HM_SIZE
        rx = int(round(fx))
        ry = int(round(fy))
        corners.append((rx, ry, blob['prob']))
    return hm_pred, labeled, corners, seg_pred

def visualize_all(img, img_resized, hm_pred, labeled, corners, fname, seg_pred=None):
    fig, axes = plt.subplots(1, 5 if seg_pred is not None else 4, figsize=(30, 6) if seg_pred is not None else (24, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(img_resized)
    axes[1].set_title("Scaled (NN input)")
    axes[1].axis('off')

    axes[2].imshow(hm_pred, cmap='hot')
    axes[2].set_title(f"Predicted Heatmap\n(threshold {HEATMAP_THRESH})")
    axes[2].axis('off')

    axes[3].imshow(img_resized)
    for (rx, ry, prob) in corners:
        axes[3].plot(rx, ry, 'go', markersize=12, label=f"{prob:.2f}")
        axes[3].annotate(f"{prob:.2f}", (rx, ry), color='lime', fontsize=12)
    axes[3].set_title(f"Predicted Corners ({len(corners)} blobs)")
    axes[3].axis('off')

    if seg_pred is not None:
        seg_up = cv2.resize(seg_pred, (img_resized.shape[1], img_resized.shape[0]), interpolation=cv2.INTER_NEAREST)
        seg_color = cv2.applyColorMap((seg_up*255).astype(np.uint8), cv2.COLORMAP_JET)
        overlay = cv2.addWeighted(cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR), 0.7, seg_color, 0.3, 0)
        axes[4].imshow(overlay)
        axes[4].set_title("Segmentation Overlay")
        axes[4].axis('off')

    plt.suptitle(f"Results for {fname}")
    plt.tight_layout()
    plt.show()

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    # model = KeypointNet()
    model = BoardKeypointNet()
    state_dict = torch.load(MODEL_PATH, map_location='cpu')
    model.load_state_dict(state_dict)
    model.eval()
    print("Model loaded.")

    exts = ['.png', '.jpg', '.jpeg', '.bmp']
    img_files = [f for f in os.listdir(IMG_FOLDER) if os.path.splitext(f)[1].lower() in exts]

    if SHOW_MAX_IMAGES is not None:
        img_files = img_files[:SHOW_MAX_IMAGES]

    print(f"Found {len(img_files)} images in {IMG_FOLDER}")

    for fname in img_files:
        img_path = os.path.join(IMG_FOLDER, fname)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Could not read {img_path}")
            continue
        img_resized = resize_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), keypoint_utils.IMG_SIZE)
        hm_pred, labeled, corners, seg_pred = predict_corners(model, img_resized)
        visualize_all(img, img_resized, hm_pred, labeled, corners, fname, seg_pred)

if __name__ == "__main__":
    main()
