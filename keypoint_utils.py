import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


IMG_SIZE = 512
HEATMAP_DOWNSAMPLE = 8
HM_SIZE = IMG_SIZE // HEATMAP_DOWNSAMPLE
# HEATMAP_SIGMA = 4
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


def _order_quad(pts):
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    assert pts.shape == (4, 2), "need 4 keypoints"
    idx = np.lexsort((pts[:, 0], pts[:, 1]))
    top2 = pts[idx[:2]]
    bot2 = pts[idx[2:]]
    tl, tr = top2[np.argsort(top2[:, 0])]
    bl, br = bot2[np.argsort(bot2[:, 0])]
    return np.stack([tl, tr, br, bl], axis=0)

# def generate_targets(kps, img_size=IMG_SIZE, hm_size=HM_SIZE):
#     assert kps.shape == (4, 2), "need 4 keypoints"
#
#     heatmap = np.zeros((hm_size, hm_size, 1), dtype=np.float32)
#     offsets = np.zeros((hm_size, hm_size, 2), dtype=np.float32)
#     offsetmask = np.zeros((hm_size, hm_size, 1), dtype=np.float32)
#     mask = np.zeros((hm_size, hm_size, 1), dtype=np.float32)
#
#     scale = hm_size / float(img_size)
#
#     # Ordered quad in heatmap coords
#     poly_hm = _order_quad(kps) * scale  # (4,2) float
#     poly_i = np.round(poly_hm).astype(np.int32)
#
#     # Fill polygon into seg_mask[:,:,0]
#     cv2.fillPoly(mask[:, :, 0], [poly_i], 1.0)
#
#     # Keypoint targets (set only the nearest cell for each keypoint)
#     for fx, fy in poly_hm:
#         nx = int(round(fx))
#         ny = int(round(fy))
#         for dx in [-1, 0, 1]:
#             for dy in [-1, 0, 1]:
#                 x = nx + dx
#                 y = ny + dy
#                 if 0 <= x < hm_size and 0 <= y < hm_size:
#                     heatmap[y, x, 0] = 1.0
#                     offsets[y, x, 0] = fx - x
#                     offsets[y, x, 1] = fy - y
#                     offsetmask[y, x, 0] = 1.0
#     #
#     # print("asdfadsgadsg")
#     # cv2.imshow("ii", heatmap)
#     # cv2.waitKey(1)
#
#     return heatmap, offsets, offsetmask, mask


## gaussian version
def generate_targets(kps, img_size=IMG_SIZE, hm_size=HM_SIZE, sigma=2.0):
    assert kps.shape == (4, 2), "need 4 keypoints"

    heatmap = np.zeros((hm_size, hm_size, 1), dtype=np.float32)
    offsets = np.zeros((hm_size, hm_size, 2), dtype=np.float32)
    offsetmask = np.zeros((hm_size, hm_size, 1), dtype=np.float32)
    mask = np.zeros((hm_size, hm_size, 1), dtype=np.float32)

    scale = hm_size / float(img_size)
    poly_hm = _order_quad(kps) * scale
    poly_i = np.round(poly_hm).astype(np.int32)
    cv2.fillPoly(mask[:, :, 0], [poly_i], 1.0)

    # Generate Gaussian for each keypoint
    for fx, fy in poly_hm:
        nx = int(round(fx))
        ny = int(round(fy))
        if 0 <= nx < hm_size and 0 <= ny < hm_size:
            # Create Gaussian blob
            y_grid, x_grid = np.ogrid[:hm_size, :hm_size]
            dist_sq = (x_grid - fx) ** 2 + (y_grid - fy) ** 2
            gaussian = np.exp(-dist_sq / (2 * sigma ** 2))

            # Max operation to handle overlapping Gaussians
            heatmap[:, :, 0] = np.maximum(heatmap[:, :, 0], gaussian)

            # Offsets only at peak
            offsets[ny, nx, 0] = max(min(fx - nx, 1.0), -1.0)
            offsets[ny, nx, 1] = max(min(fy - ny, 1.0), -1.0)
            offsetmask[ny, nx, 0] = 1.0

    return heatmap, offsets, offsetmask, mask

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


def visualize_targets(img, hm, off, kps, seg_mask, title_prefix=""):
    fig, axes = plt.subplots(1, 4, figsize=(24, 6))

    # input + gt corners
    axes[0].imshow(img)
    for kp in kps:
        axes[0].plot(kp[0], kp[1], 'rx', markersize=10)
    axes[0].set_title(f"{title_prefix}Input w/ GT corners")

    # heatmap
    axes[1].imshow(hm[..., 0], cmap='hot')
    axes[1].set_title(f"{title_prefix}Target Heatmap")

    # reconstruction
    axes[2].imshow(img)
    rec_kps = reconstruct_keypoints(hm, off, threshold=0.9)
    for rx, ry in rec_kps:
        axes[2].plot(rx, ry, 'go')
    axes[2].set_title(f"{title_prefix}Reconstructed (hm>0.9) pts")

    # segmentation mask overlay
    axes[3].imshow(img, alpha=0.7)
    axes[3].imshow(seg_mask[..., 0], cmap='jet', alpha=0.4)
    axes[3].set_title(f"{title_prefix}Segmentation Mask")

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


def test_generate_and_reconstruct():
    """
    Automated test: generate targets from synthetic keypoints, reconstruct them, and check accuracy.
    """
    print("Running automated test: generate_targets -> reconstruct_keypoints")
    # Synthetic quad (square in image coordinates)
    kps = np.array([
        [50, 50],    # top-left
        [462, 50],   # top-right
        [462, 462],  # bottom-right
        [50, 462]    # bottom-left
    ], dtype=np.float32)
    hm, off, _, _ = generate_targets(kps)
    rec_kps = reconstruct_keypoints(hm, off, threshold=0.5)
    # Order reconstructed points for comparison
    rec_kps = np.array(rec_kps, dtype=np.float32)
    if len(rec_kps) != 4:
        print(f"FAIL: Expected 4 keypoints, got {len(rec_kps)}")
        return False
    # Order both sets for comparison
    kps_ordered = _order_quad(kps)
    rec_ordered = _order_quad(rec_kps)
    # Compare positions
    errors = np.linalg.norm(kps_ordered - rec_ordered, axis=1)
    print(f"Keypoint errors: {errors}")
    tol = 1.0  # pixel tolerance
    if np.all(errors < tol):
        print("PASS: All reconstructed keypoints are within tolerance.")
        return True
    else:
        print("FAIL: Some keypoints differ by more than tolerance.")
        return False

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
        hm, off, _, mask = generate_targets(kps_rescaled)
        visualize_targets(img_resized, hm, off, kps_rescaled, mask, title_prefix=f"Sample {i+1}: ")
        visualize_offsets(img_resized, hm, off, title_prefix=f"Sample {i+1}: ")

if __name__ == "__main__":
    test_generate_and_reconstruct()
    main()
