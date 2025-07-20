import os
import numpy as np
import cv2
import matplotlib.pyplot as plt

IMG_SIZE = 1000
HEATMAP_DOWNSAMPLE = 4
HM_SIZE = IMG_SIZE // HEATMAP_DOWNSAMPLE
HEATMAP_SIGMA = 4
NUM_CORNERS = 4


def parse_annotations(ann_path, files_folder):
    images = []
    keypoints = []
    with open(ann_path) as f:
        for line in f:
            parts = line.strip().split(',')
            fname = parts[0].strip()
            coords = list(map(int, parts[1:]))
            kps = np.array(coords).reshape(NUM_CORNERS, 2)
            img_path = os.path.join(files_folder, fname)
            images.append(img_path)
            keypoints.append(kps)
    return images, keypoints


def resize_img_and_kps(img, kps, desired_size=IMG_SIZE):
    h0, w0 = img.shape[:2]
    scale_x = desired_size / w0
    scale_y = desired_size / h0
    img_resized = cv2.resize(img, (desired_size, desired_size))
    kps_rescaled = kps.astype(np.float32)
    kps_rescaled[:, 0] *= scale_x
    kps_rescaled[:, 1] *= scale_y
    return img_resized, kps_rescaled


def generate_targets(kps, img_size=IMG_SIZE, hm_size=HM_SIZE):
    heatmap = np.zeros((hm_size, hm_size, 1), dtype=np.float32)
    offsets = np.zeros((hm_size, hm_size, 2), dtype=np.float32)
    mask = np.zeros((hm_size, hm_size, 1), dtype=np.float32)
    scale = hm_size / img_size

    for kp in kps:
        x, y = kp
        fx = x * scale
        fy = y * scale
        ix, iy = int(round(fx)), int(round(fy))
        if ix < 0 or iy < 0 or ix >= hm_size or iy >= hm_size:
            continue

        for dx in range(-3 * HEATMAP_SIGMA, 3 * HEATMAP_SIGMA + 1):
            for dy in range(-3 * HEATMAP_SIGMA, 3 * HEATMAP_SIGMA + 1):
                nx, ny = ix + dx, iy + dy
                if not (0 <= nx < hm_size and 0 <= ny < hm_size):
                    continue

                d2 = (fx - nx) ** 2 + (fy - ny) ** 2
                val = np.exp(-d2 / (2 * HEATMAP_SIGMA ** 2))

                if val > heatmap[ny, nx, 0]:
                    heatmap[ny, nx, 0] = val
                    offsets[ny, nx, 0] = fx - nx
                    offsets[ny, nx, 1] = fy - ny
                    mask[ny, nx, 0] = 1

    return heatmap, offsets, mask


def reconstruct_keypoints(heatmap, offsets, threshold=0.5, img_size=IMG_SIZE, hm_size=HM_SIZE):
    """
    Decode heatmap and offsets into image-space keypoints.
    Only considers heatmap cells with value > threshold.
    """
    scale = hm_size / img_size
    ys, xs = np.where(heatmap[..., 0] > threshold)
    kps = []
    for y, x in zip(ys, xs):
        fx_hat = x + offsets[y, x, 0]
        fy_hat = y + offsets[y, x, 1]
        x_rec = fx_hat / scale
        y_rec = fy_hat / scale
        kps.append((x_rec, y_rec))
    return kps


def visualize_targets(img, hm, off, kps, title_prefix=""):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    axes[0].imshow(img)
    for kp in kps:
        axes[0].plot(kp[0], kp[1], 'rx', markersize=10)
    axes[0].set_title(f"{title_prefix}Input w/ GT corners")

    axes[1].imshow(hm[..., 0], cmap='hot')
    axes[1].set_title(f"{title_prefix}Target Heatmap")

    axes[2].imshow(img)
    rec_kps = reconstruct_keypoints(hm, off, threshold=0.9)
    for rx, ry in rec_kps:
        axes[2].plot(rx, ry, 'go')
    axes[2].set_title(f"{title_prefix}Reconstructed (hm>0.9) pts")
    plt.show()


def visualize_offsets(img, hm, off, title_prefix=""):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(hm[..., 0], cmap='hot')
    axes[0].set_title(f"{title_prefix}Heatmap")

    axes[1].imshow(img)
    rec_kps = reconstruct_keypoints(hm, off, threshold=0.5)
    for rx, ry in rec_kps:
        axes[1].plot(rx, ry, 'go')
    axes[1].set_title(f"{title_prefix}Reconstructed (hm>0.5) pts ({len(rec_kps)} found)")
    plt.show()


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Visualize keypoint dataset for board detection.')
    parser.add_argument('--annotations', type=str, default='annotations.txt', help='Path to annotations.txt')
    parser.add_argument('--images', type=str, default='files', help='Folder with game board images')
    parser.add_argument('--samples', type=int, default=3, help='Number of samples to visualize')
    args = parser.parse_args()

    img_paths, keypoints = parse_annotations(args.annotations, args.images)
    for i in range(min(args.samples, len(img_paths))):
        img = cv2.imread(img_paths[i])
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized, kps_rescaled = resize_img_and_kps(img_rgb, keypoints[i], IMG_SIZE)
        hm, off, _ = generate_targets(kps_rescaled)
        visualize_targets(img_resized, hm, off, kps_rescaled, title_prefix=f"Sample {i+1}: ")
        visualize_offsets(img_resized, hm, off, title_prefix=f"Sample {i+1}: ")

if __name__ == "__main__":
    main()
