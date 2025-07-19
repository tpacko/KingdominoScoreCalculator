import os
import numpy as np
import cv2
import tensorflow as tf
import matplotlib.pyplot as plt
from scipy.ndimage import label

# ==== USER-EDITABLE PARAMETERS (set all here!) ====
MODEL_PATH      = "board_keypoint_detector_v1.h5"
IMG_FOLDER      = "files"
IMG_SIZE        = 1000
HEATMAP_DOWNSAMPLE = 8
NUM_CORNERS     = 4
HEATMAP_THRESH  = 0.95   # Minimum probability for blob detection
SHOW_MAX_IMAGES = 50
# ================================================

HM_SIZE = IMG_SIZE // HEATMAP_DOWNSAMPLE

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
    inp = img_resized / 255.0
    inp = np.expand_dims(inp, 0)
    hm_pred, off_pred = model.predict(inp)
    hm_pred = hm_pred[0, ..., 0]
    off_pred = off_pred[0]

    labeled, num = blobs_from_heatmap(hm_pred, min_prob=HEATMAP_THRESH)
    blobs = get_blob_peaks(labeled, num, hm_pred)

    corners = []
    for blob in blobs[:NUM_CORNERS]:
        x, y = blob['peak_x'], blob['peak_y']
        fx = x * IMG_SIZE / HM_SIZE
        fy = y * IMG_SIZE / HM_SIZE
        ox, oy = off_pred[y, x]
        rx, ry = fx + ox, fy + oy
        corners.append((rx, ry, blob['prob']))
    return hm_pred, labeled, corners

def visualize_all(img, img_resized, hm_pred, labeled, corners, fname):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))
    axes[0].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("Original Image")
    axes[0].axis('off')

    axes[1].imshow(img_resized)
    axes[1].set_title("Scaled (NN input)")
    axes[1].axis('off')

    axes[2].imshow(hm_pred, cmap='hot')
    axes[2].set_title(f"Predicted Heatmap\n(threshold {HEATMAP_THRESH})")
    axes[2].axis('off')

    # Mark all blobs' peaks on the image
    axes[3].imshow(img_resized)
    for (rx, ry, prob) in corners:
        axes[3].plot(rx, ry, 'go', markersize=12, label=f"{prob:.2f}")
        axes[3].annotate(f"{prob:.2f}", (rx, ry), color='lime', fontsize=12)
    axes[3].set_title(f"Predicted Corners ({len(corners)} blobs)")
    axes[3].axis('off')
    plt.suptitle(f"Results for {fname}")
    plt.tight_layout()
    plt.show()

def main():
    print(f"Loading model from {MODEL_PATH} ...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
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
        img_resized = resize_img(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), IMG_SIZE)
        hm_pred, labeled, corners = predict_corners(model, img_resized)
        visualize_all(img, img_resized, hm_pred, labeled, corners, fname)

if __name__ == "__main__":
    main()
