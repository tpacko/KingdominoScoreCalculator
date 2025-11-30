import numpy as np
import cv2

# Augmentation functions

def identity(img, kps):
    return img, kps

def rotate_90(img, kps):
    img_size = img.shape[0]
    img_rot = np.ascontiguousarray(np.rot90(img, 1))
    kps_rot = np.stack([kps[:, 1], img_size - 1 - kps[:, 0]], axis=-1)
    return img_rot, kps_rot

def rotate_180(img, kps):
    img_size = img.shape[0]
    img_rot = np.ascontiguousarray(np.rot90(img, 2))
    kps_rot = np.stack([img_size - 1 - kps[:, 0], img_size - 1 - kps[:, 1]], axis=-1)
    return img_rot, kps_rot

def rotate_neg90(img, kps):
    img_size = img.shape[0]
    img_rot = np.ascontiguousarray(np.rot90(img, -1))
    kps_rot = np.stack([img_size - 1 - kps[:, 1], kps[:, 0]], axis=-1)
    return img_rot, kps_rot

def keypoints_in_bounds(kps, size):
    return not (np.any(kps < 0) or np.any(kps >= size))

def horizontal_flip(img, kps):
    img_size = img.shape[0]
    img_flipped = np.fliplr(img)
    kps_flipped = kps.copy()
    kps_flipped[:, 0] = img_size - 1 - kps[:, 0]
    if not keypoints_in_bounds(kps_flipped, img_size):
        return img, kps
    return img_flipped, kps_flipped

def vertical_flip(img, kps):
    img_size = img.shape[0]
    img_flipped = np.flipud(img)
    kps_flipped = kps.copy()
    kps_flipped[:, 1] = img_size - 1 - kps[:, 1]
    if not keypoints_in_bounds(kps_flipped, img_size):
        return img, kps
    return img_flipped, kps_flipped

def random_small_rotation(img, kps, angle_range=30):
    img_size = img.shape[0]
    angle = np.random.uniform(-angle_range, angle_range)
    center = (img_size/2, img_size/2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img_rot = cv2.warpAffine(img, M, (img_size, img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    kps_xy = kps[..., :2]
    ones = np.ones((kps.shape[0], 1))
    kps_homo = np.concatenate([kps_xy, ones], axis=1)
    kps_rot_xy = (M @ kps_homo.T).T
    # If there are extra columns, concatenate them back
    if kps.shape[1] > 2:
        kps_rot = np.concatenate([kps_rot_xy, kps[..., 2:]], axis=1)
    else:
        kps_rot = kps_rot_xy
    if not keypoints_in_bounds(kps_rot[..., :2], img_size):
        return img, kps
    return img_rot, kps_rot

def random_scale(img, kps, scale_range=(0.9, 1.2)):
    img_size = img.shape[0]
    scale = np.random.uniform(*scale_range)
    M = np.array([
        [scale, 0, (1-scale)*img_size/2],
        [0, scale, (1-scale)*img_size/2]
    ])
    img_scaled = cv2.warpAffine(img, M, (img_size, img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    kps_xy = kps[..., :2]
    ones = np.ones((kps.shape[0], 1))
    kps_homo = np.concatenate([kps_xy, ones], axis=1)
    kps_scaled_xy = (M @ kps_homo.T).T
    if kps.shape[1] > 2:
        kps_scaled = np.concatenate([kps_scaled_xy, kps[..., 2:]], axis=1)
    else:
        kps_scaled = kps_scaled_xy
    if not keypoints_in_bounds(kps_scaled[..., :2], img_size):
        return img, kps
    return img_scaled, kps_scaled

def random_translate(img, kps, frac=0.1):
    img_size = img.shape[0]
    tx = np.random.uniform(-img_size*frac, img_size*frac)
    ty = np.random.uniform(-img_size*frac, img_size*frac)
    M = np.array([
        [1, 0, tx],
        [0, 1, ty]
    ])
    img_trans = cv2.warpAffine(img, M, (img_size, img_size), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    kps_trans = kps.copy()
    kps_trans[..., 0] += tx
    kps_trans[..., 1] += ty
    if not keypoints_in_bounds(kps_trans[..., :2], img_size):
        return img, kps
    return img_trans, kps_trans

def color_jitter(img, kps, brightness=0.5, contrast=0.5, saturation=0.5):
    img_out = img.astype(np.float32)
    img_out += np.random.uniform(-brightness*255, brightness*255)
    factor = np.random.uniform(1-contrast, 1+contrast)
    img_out = 127.5 + factor * (img_out - 127.5)
    img_hsv = cv2.cvtColor(np.clip(img_out, 0, 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat_factor = np.random.uniform(1-saturation, 1+saturation)
    img_hsv[...,1] = np.clip(img_hsv[...,1]*sat_factor, 0, 255)
    img_out = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
    return img_out.astype(np.uint8), kps

def random_perspective(img, kps, max_warp=0.05):
    img_size = img.shape[0]
    margin = img_size * max_warp
    src = np.array([
        [0, 0],
        [img_size-1, 0],
        [img_size-1, img_size-1],
        [0, img_size-1]
    ], dtype=np.float32)
    dst = src + np.random.uniform(-margin, margin, src.shape).astype(np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    img_warp = cv2.warpPerspective(img, M, (img_size, img_size), borderMode=cv2.BORDER_CONSTANT)
    kps_xy = kps[..., :2]
    kps_homo = np.concatenate([kps_xy, np.ones((kps.shape[0], 1))], axis=1)
    kps_transformed = (M @ kps_homo.T).T
    kps_warp_xy = kps_transformed[:, :2] / kps_transformed[:, 2:3]
    if kps.shape[1] > 2:
        kps_warp = np.concatenate([kps_warp_xy, kps[..., 2:]], axis=1)
    else:
        kps_warp = kps_warp_xy
    if not keypoints_in_bounds(kps_warp[..., :2], img_size):
        return img, kps
    return img_warp, kps_warp

def switch_color_channels(img, kps):
    # 1. Create the permutation indices (0, 1, 2)
    indices = np.random.permutation(3)

    # 2. Reorder the image array using these indices along the last axis
    shuffled_img = img[:, :, indices]
    return shuffled_img, kps

def blur(img, kps, kernel_size = 5):
    return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0), kps

def random_noise(img, kps, mean=0, std=10):
    noise = np.random.normal(mean, std, img.shape).astype(np.float32)
    noisy_img = np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)
    return noisy_img, kps

def draw_keypoints(img, kps, color=(0,255,0)):
    img_disp = img.copy()
    for kp in kps:
        cv2.circle(img_disp, (int(kp[0]), int(kp[1])), 5, color, -1)
    return img_disp

def test_augmentation_correctness():
    img_size = 100
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), -1)
    kps = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)

    augmentations = [
        (identity, 'identity'),
        (rotate_90, 'rotate_90'),
        (rotate_180, 'rotate_180'),
        (rotate_neg90, 'rotate_neg90'),
        (horizontal_flip, 'horizontal_flip'),
        (vertical_flip, 'vertical_flip'),
        (random_small_rotation, 'random_small_rotation'),
        (random_scale, 'random_scale'),
        (random_translate, 'random_translate'),
        (color_jitter, 'color_jitter'),
        (random_perspective, 'random_perspective'),
        (switch_color_channels, 'switch_color_channels'),
        (blur, 'blur'),
        (random_noise, 'random_noise'),
    ]

    results = []
    for aug_fn, name in augmentations:
        aug_img, aug_kps = aug_fn(img, kps)
        # Test 1: Output image shape matches input
        shape_ok = aug_img.shape == img.shape
        # Test 2: Keypoints within bounds
        kps_ok = np.all((aug_kps >= 0) & (aug_kps < img_size))
        # Test 3: For deterministic augmentations, check expected keypoints
        expected = None
        if name == 'identity':
            expected = np.allclose(aug_kps, kps)
        elif name == 'rotate_90':
            expected_kps = np.stack([kps[:, 1], img_size - 1 - kps[:, 0]], axis=-1)
            expected = np.allclose(aug_kps, expected_kps)
        elif name == 'rotate_180':
            expected_kps = np.stack([img_size - 1 - kps[:, 0], img_size - 1 - kps[:, 1]], axis=-1)
            expected = np.allclose(aug_kps, expected_kps)
        elif name == 'rotate_neg90':
            expected_kps = np.stack([img_size - 1 - kps[:, 1], kps[:, 0]], axis=-1)
            expected = np.allclose(aug_kps, expected_kps)
        elif name == 'horizontal_flip':
            expected_kps = kps.copy()
            expected_kps[:, 0] = img_size - 1 - kps[:, 0]
            expected = np.allclose(aug_kps, expected_kps)
        elif name == 'vertical_flip':
            expected_kps = kps.copy()
            expected_kps[:, 1] = img_size - 1 - kps[:, 1]
            expected = np.allclose(aug_kps, expected_kps)
        # For random augmentations, expected is None
        results.append((name, shape_ok, kps_ok, expected))
    return results


def main():
    # Create a synthetic image (100x100 RGB) and keypoints
    img_size = 100
    img = np.ones((img_size, img_size, 3), dtype=np.uint8) * 200
    # Draw a red square
    cv2.rectangle(img, (20, 20), (80, 80), (0, 0, 255), -1)
    # Keypoints: corners of the square
    kps = np.array([[20, 20], [80, 20], [80, 80], [20, 80]], dtype=np.float32)

    augmentations = [
        (identity, 'identity'),
        (rotate_90, 'rotate_90'),
        (rotate_180, 'rotate_180'),
        (rotate_neg90, 'rotate_neg90'),
        (horizontal_flip, 'horizontal_flip'),
        (vertical_flip, 'vertical_flip'),
        (random_small_rotation, 'random_small_rotation'),
        (random_scale, 'random_scale'),
        (random_translate, 'random_translate'),
        (color_jitter, 'color_jitter'),
        (random_perspective, 'random_perspective'),
        (switch_color_channels, 'switch_color_channels'),
        (blur, 'blur'),
        (random_noise, 'random_noise'),
    ]

    print("Testing augmentations on synthetic image and keypoints...")
    for aug_fn, name in augmentations:
        aug_img, aug_kps = aug_fn(img, kps)
        print(f"{name}: img shape={aug_img.shape}, kps={aug_kps}")
        img_with_kps = draw_keypoints(aug_img, aug_kps)
        cv2.imwrite(f"test_{name}.png", img_with_kps)
    print("Augmentation tests complete. Check test_*.png files for results.")

    # Automated correctness tests
    print("\nRunning automated augmentation correctness tests...")
    results = test_augmentation_correctness()
    for name, shape_ok, kps_ok, expected in results:
        print(f"{name:22} | shape ok: {shape_ok} | kps in bounds: {kps_ok} | expected transform: {expected}")
    print("Automated tests complete.")

if __name__ == "__main__":
    main()
