import numpy as np
import cv2
import matplotlib.pyplot as plt
import math

# ============================================================================
# CONFIGURATION
# ============================================================================
# IMAGE_PATH = 'boards/game1_board.png'
IMAGE_PATH = 'boards/game10_board.jpg'

# Preprocessing options (applied in order)
PREPROCESS_BLUR = True
BLUR_KERNEL = (9, 9)
BLUR_SIGMA = 0

APPLY_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = (8, 8)

# Line filtering (core step for grid detection)
USE_HOUGH_LINE_FILTER = True  # Detect lines using Canny + HoughLinesP
USE_MORPHOLOGICAL_FILTER = False  # Morphological opening with h/v kernels

# Hough line detection parameters
CANNY_THRESH1 = 150
CANNY_THRESH2 = 250
HOUGH_THRESHOLD = 80
MIN_LINE_LENGTH = 50
MAX_LINE_GAP = 10
ANGLE_TOLERANCE = 10  # degrees from perfect h/v
LINE_DRAW_THICKNESS = 3

# Morphological filter parameters
MORPH_KERNEL_SIZE = 15  # kernel length for line extraction

# Optional post-processing (usually not needed after line filtering)
APPLY_BINARY_THRESH = False
APPLY_CANNY_FINAL = False

# FFT display options
LOG_SCALE = True
SHIFT_ZERO_FREQ = True
COLORMAP = 'gray'


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def load_image(path):
    """Load and convert image to grayscale."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not load image from {path}")
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return gray


def apply_blur(img):
    """Apply Gaussian blur."""
    if PREPROCESS_BLUR:
        return cv2.GaussianBlur(img, BLUR_KERNEL, BLUR_SIGMA)
    return img


def apply_clahe(img):
    """Apply CLAHE contrast enhancement."""
    if APPLY_CLAHE:
        clahe = cv2.createCLAHE(clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE)
        return clahe.apply(img)
    return img


def hough_line_filter(img):
    """Extract straight horizontal and vertical lines using Canny + HoughLinesP."""
    edges = cv2.Canny(img, CANNY_THRESH1, CANNY_THRESH2)

    # Adjust min line length based on image size
    min_len = max(MIN_LINE_LENGTH, min(img.shape) // 20)

    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, HOUGH_THRESHOLD,
                            minLineLength=min_len, maxLineGap=MAX_LINE_GAP)

    h_mask = np.zeros_like(img, dtype=np.uint8)
    v_mask = np.zeros_like(img, dtype=np.uint8)

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx == 0 and dy == 0:
                continue

            angle = abs(math.degrees(math.atan2(dy, dx)))

            # Horizontal: angle ~0 or ~180
            if angle <= ANGLE_TOLERANCE or abs(angle - 180) <= ANGLE_TOLERANCE:
                cv2.line(h_mask, (x1, y1), (x2, y2), 255, LINE_DRAW_THICKNESS)
            # Vertical: angle ~90
            elif abs(angle - 90) <= ANGLE_TOLERANCE:
                cv2.line(v_mask, (x1, y1), (x2, y2), 255, LINE_DRAW_THICKNESS)

    # Dilate slightly to connect gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    h_mask = cv2.dilate(h_mask, kernel, iterations=1)
    v_mask = cv2.dilate(v_mask, kernel, iterations=1)

    # Combine masks
    combined = cv2.bitwise_or(h_mask, v_mask)
    return combined, h_mask, v_mask


def morphological_line_filter(img):
    """Extract lines using morphological opening with directional kernels."""
    # Horizontal kernel
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (MORPH_KERNEL_SIZE, 1))
    h_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, h_kernel)

    # Vertical kernel
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, MORPH_KERNEL_SIZE))
    v_lines = cv2.morphologyEx(img, cv2.MORPH_OPEN, v_kernel)

    # Combine
    combined = cv2.bitwise_or(h_lines, v_lines)
    return combined, h_lines, v_lines


def preprocess_image(img):
    """
    Main preprocessing pipeline. Returns preprocessed image and debug masks.

    Order: Grayscale → Blur → CLAHE → Line Filtering → Optional Threshold/Canny
    """
    # Step 1: Ensure uint8 grayscale
    processed = img.copy()
    if processed.dtype != np.uint8:
        processed = np.clip(processed, 0, 255).astype(np.uint8)

    # Step 2: Blur (noise reduction)
    processed = apply_blur(processed)

    # Step 3: CLAHE (contrast enhancement)
    processed = apply_clahe(processed)

    # Step 4: Line filtering (CORE STEP)
    h_mask = v_mask = None
    if USE_HOUGH_LINE_FILTER and USE_MORPHOLOGICAL_FILTER:
        # Use both methods - combine results
        hough_result, h_hough, v_hough = hough_line_filter(processed)
        morph_result, h_morph, v_morph = morphological_line_filter(processed)
        processed = cv2.bitwise_or(hough_result, morph_result)
        h_mask = cv2.bitwise_or(h_hough, h_morph)
        v_mask = cv2.bitwise_or(v_hough, v_morph)
    elif USE_HOUGH_LINE_FILTER:
        processed, h_mask, v_mask = hough_line_filter(processed)
    elif USE_MORPHOLOGICAL_FILTER:
        processed, h_mask, v_mask = morphological_line_filter(processed)

    # Step 5: Optional binary threshold
    if APPLY_BINARY_THRESH:
        _, processed = cv2.threshold(processed, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 6: Optional Canny (usually not needed)
    if APPLY_CANNY_FINAL:
        processed = cv2.Canny(processed, CANNY_THRESH1, CANNY_THRESH2)

    return processed, h_mask, v_mask


def compute_fft_2d(image):
    """Compute 2D FFT."""
    fft = np.fft.fft2(image)
    if SHIFT_ZERO_FREQ:
        fft = np.fft.fftshift(fft)
    return fft


def compute_fft_1d_x(image):
    """Compute 1D FFT along X axis (averaged over Y)."""
    signal_x = np.mean(image, axis=0)
    fft_x = np.fft.fft(signal_x)
    if SHIFT_ZERO_FREQ:
        fft_x = np.fft.fftshift(fft_x)
    return fft_x


def compute_fft_1d_y(image):
    """Compute 1D FFT along Y axis (averaged over X)."""
    signal_y = np.mean(image, axis=1)
    fft_y = np.fft.fft(signal_y)
    if SHIFT_ZERO_FREQ:
        fft_y = np.fft.fftshift(fft_y)
    return fft_y


def get_magnitude_spectrum(fft_result):
    """Convert FFT to magnitude spectrum."""
    magnitude = np.abs(fft_result)
    if LOG_SCALE:
        magnitude = np.log(magnitude + 1)
    return magnitude



def find_nearest_index(arr, value):
    """Find index of nearest value in array."""
    return int(np.argmin(np.abs(arr - value)))


# ============================================================================
# MAIN ANALYSIS
# ============================================================================

def main():
    print(f"Loading image: {IMAGE_PATH}")
    original = load_image(IMAGE_PATH)
    height, width = original.shape
    print(f"Image dimensions: {width}x{height}")

    # Preprocess
    print("Preprocessing...")
    preprocessed, h_mask, v_mask = preprocess_image(original)

    # Compute FFTs
    print("Computing FFTs...")
    fft_2d = compute_fft_2d(preprocessed)
    magnitude_2d = get_magnitude_spectrum(fft_2d)

    fft_x = compute_fft_1d_x(preprocessed)
    fft_y = compute_fft_1d_y(preprocessed)
    magnitude_x = get_magnitude_spectrum(fft_x)
    magnitude_y = get_magnitude_spectrum(fft_y)

    # Convert to spacing domain
    freq_x = np.fft.fftfreq(len(magnitude_x))
    freq_y = np.fft.fftfreq(len(magnitude_y))
    if SHIFT_ZERO_FREQ:
        freq_x = np.fft.fftshift(freq_x)
        freq_y = np.fft.fftshift(freq_y)

    spacing_x = np.zeros_like(freq_x)
    spacing_y = np.zeros_like(freq_y)
    mask_x = freq_x != 0
    mask_y = freq_y != 0
    spacing_x[mask_x] = 1.0 / np.abs(freq_x[mask_x])
    spacing_y[mask_y] = 1.0 / np.abs(freq_y[mask_y])

    # Focus on valid spacing range (50-500 pixels)
    valid_x = (spacing_x >= 50) & (spacing_x <= 500)
    valid_y = (spacing_y >= 50) & (spacing_y <= 500)

    spacing_vals_x = spacing_x[valid_x]
    spacing_vals_y = spacing_y[valid_y]
    mags_x = magnitude_x[valid_x]
    mags_y = magnitude_y[valid_y]

    # Target spacings for 8 lines (7 intervals) vs 6 lines (5 intervals)
    target_7_x = width / 7.0
    target_7h_x = width / 14.0  # half-frequency harmonic
    target_5_x = width / 5.0
    target_5h_x = width / 10.0

    target_7_y = height / 7.0
    target_7h_y = height / 14.0
    target_5_y = height / 5.0
    target_5h_y = height / 10.0

    # Analyze X-axis (spatial domain)
    print("\nAnalyzing X-axis spacing...")
    results_x = {}
    if len(spacing_vals_x) > 0:
        idx_7 = find_nearest_index(spacing_vals_x, target_7_x)
        idx_7h = find_nearest_index(spacing_vals_x, target_7h_x)
        idx_5 = find_nearest_index(spacing_vals_x, target_5_x)
        idx_5h = find_nearest_index(spacing_vals_x, target_5h_x)

        # Get magnitudes directly at target frequencies
        mag_7 = mags_x[idx_7]
        mag_7h = mags_x[idx_7h]
        mag_5 = mags_x[idx_5]
        mag_5h = mags_x[idx_5h]

        # Combine using mean of fundamental + harmonic
        combined_7 = (mag_7 + mag_7h) / 2.0
        combined_5 = (mag_5 + mag_5h) / 2.0

        results_x = {
            'idx_7': idx_7, 'idx_7h': idx_7h, 'idx_5': idx_5, 'idx_5h': idx_5h,
            'mag_7': mag_7, 'mag_7h': mag_7h, 'mag_5': mag_5, 'mag_5h': mag_5h,
            'combined_7': combined_7, 'combined_5': combined_5,
            'winner': '8 lines (7 intervals)' if combined_7 > combined_5 else '6 lines (5 intervals)',
            'ratio': max(combined_7, combined_5) / min(combined_7, combined_5) if min(combined_7,
                                                                                      combined_5) > 0 else float('inf')
        }

    # Analyze Y-axis (spatial domain)
    print("Analyzing Y-axis spacing...")
    results_y = {}
    if len(spacing_vals_y) > 0:
        idx_7 = find_nearest_index(spacing_vals_y, target_7_y)
        idx_7h = find_nearest_index(spacing_vals_y, target_7h_y)
        idx_5 = find_nearest_index(spacing_vals_y, target_5_y)
        idx_5h = find_nearest_index(spacing_vals_y, target_5h_y)

        # Get magnitudes directly at target frequencies
        mag_7 = mags_y[idx_7]
        mag_7h = mags_y[idx_7h]
        mag_5 = mags_y[idx_5]
        mag_5h = mags_y[idx_5h]

        # Combine using mean of fundamental + harmonic
        combined_7 = (mag_7 + mag_7h) / 2.0
        combined_5 = (mag_5 + mag_5h) / 2.0

        results_y = {
            'idx_7': idx_7, 'idx_7h': idx_7h, 'idx_5': idx_5, 'idx_5h': idx_5h,
            'mag_7': mag_7, 'mag_7h': mag_7h, 'mag_5': mag_5, 'mag_5h': mag_5h,
            'combined_7': combined_7, 'combined_5': combined_5,
            'winner': '8 lines (7 intervals)' if combined_7 > combined_5 else '6 lines (5 intervals)',
            'ratio': max(combined_7, combined_5) / min(combined_7, combined_5) if min(combined_7,
                                                                                      combined_5) > 0 else float('inf')
        }

    # ========================================================================
    # VISUALIZATION
    # ========================================================================
    fig = plt.figure(figsize=(18, 12))

    # Plot 1: Original image
    ax1 = plt.subplot(2, 3, 1)
    ax1.imshow(original, cmap='gray')
    ax1.set_title('Original Image')
    ax1.axis('off')

    # Plot 2: Preprocessed image
    ax2 = plt.subplot(2, 3, 2)
    ax2.imshow(preprocessed, cmap='gray')
    ax2.set_title('Preprocessed (Line Filtered)')
    ax2.axis('off')

    # Plot 3: 2D FFT magnitude
    ax3 = plt.subplot(2, 3, 3)
    im3 = ax3.imshow(magnitude_2d, cmap=COLORMAP)
    ax3.set_title('2D FFT Magnitude' + (' (Log)' if LOG_SCALE else ''))
    plt.colorbar(im3, ax=ax3)

    # Plot 4: Spatial X
    ax4 = plt.subplot(2, 3, 4)
    if len(spacing_vals_x) > 0:
        ax4.plot(spacing_vals_x, mags_x, linewidth=1.5, color='steelblue')

        # Mark target spacings
        ax4.axvline(target_7_x, color='red', linestyle='--', alpha=0.7, label=f'{target_7_x:.1f}px (7 int)')
        ax4.axvline(target_7h_x, color='orange', linestyle=':', alpha=0.7, label=f'{target_7h_x:.1f}px (7 harm)')
        ax4.axvline(target_5_x, color='green', linestyle='--', alpha=0.7, label=f'{target_5_x:.1f}px (5 int)')
        ax4.axvline(target_5h_x, color='lime', linestyle=':', alpha=0.7, label=f'{target_5h_x:.1f}px (5 harm)')

        # Mark detected peaks
        ax4.plot(spacing_vals_x[results_x['idx_7']], mags_x[results_x['idx_7']], 'ro', markersize=8)
        ax4.plot(spacing_vals_x[results_x['idx_7h']], mags_x[results_x['idx_7h']], 'o', color='orangered', markersize=6)
        ax4.plot(spacing_vals_x[results_x['idx_5']], mags_x[results_x['idx_5']], 'go', markersize=8)
        ax4.plot(spacing_vals_x[results_x['idx_5h']], mags_x[results_x['idx_5h']], 'o', color='limegreen', markersize=6)

        ax4.text(0.02, 0.98, f"WINNER: {results_x['winner']}\nRatio: {results_x['ratio']:.2f}x",
                 transform=ax4.transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax4.set_title('Spatial Domain X (spacing)')
    ax4.set_xlabel('Spacing (pixels)')
    ax4.set_ylabel('Magnitude')
    ax4.set_xlim(50, 500)
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=7)

    # Plot 5: Spatial Y
    ax5 = plt.subplot(2, 3, 5)
    if len(spacing_vals_y) > 0:
        ax5.plot(spacing_vals_y, mags_y, linewidth=1.5, color='steelblue')

        ax5.axvline(target_7_y, color='red', linestyle='--', alpha=0.7, label=f'{target_7_y:.1f}px (7 int)')
        ax5.axvline(target_7h_y, color='orange', linestyle=':', alpha=0.7, label=f'{target_7h_y:.1f}px (7 harm)')
        ax5.axvline(target_5_y, color='green', linestyle='--', alpha=0.7, label=f'{target_5_y:.1f}px (5 int)')
        ax5.axvline(target_5h_y, color='lime', linestyle=':', alpha=0.7, label=f'{target_5h_y:.1f}px (5 harm)')

        ax5.plot(spacing_vals_y[results_y['idx_7']], mags_y[results_y['idx_7']], 'ro', markersize=8)
        ax5.plot(spacing_vals_y[results_y['idx_7h']], mags_y[results_y['idx_7h']], 'o', color='orangered', markersize=6)
        ax5.plot(spacing_vals_y[results_y['idx_5']], mags_y[results_y['idx_5']], 'go', markersize=8)
        ax5.plot(spacing_vals_y[results_y['idx_5h']], mags_y[results_y['idx_5h']], 'o', color='limegreen', markersize=6)

        ax5.text(0.02, 0.98, f"WINNER: {results_y['winner']}\nRatio: {results_y['ratio']:.2f}x",
                 transform=ax5.transAxes, fontsize=10, va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    ax5.set_title('Spatial Domain Y (spacing)')
    ax5.set_xlabel('Spacing (pixels)')
    ax5.set_ylabel('Magnitude')
    ax5.set_xlim(50, 500)
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=7)

    # Plot 6: Line detection masks (bonus)
    ax6 = plt.subplot(2, 3, 6)
    if h_mask is not None and v_mask is not None:
        overlay = cv2.cvtColor(original, cv2.COLOR_GRAY2BGR)
        overlay[h_mask > 0] = [255, 0, 0]  # Red for horizontal
        overlay[v_mask > 0] = [0, 255, 0]  # Green for vertical
        ax6.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax6.set_title('Detected Lines (R=horiz, G=vert)')
    else:
        ax6.text(0.5, 0.5, 'No line detection used', ha='center', va='center', transform=ax6.transAxes)
        ax6.set_title('Line Detection Overlay')
    ax6.axis('off')

    plt.tight_layout()
    output_file = 'fft_analysis_output.png'
    plt.savefig(output_file, dpi=150)
    print(f"\nSaved visualization: {output_file}")

    # Show the plot
    plt.show()

    # ========================================================================
    # PRINT RESULTS
    # ========================================================================
    print("\n" + "=" * 70)
    print("SPATIAL DOMAIN ANALYSIS RESULTS")
    print("=" * 70)

    if results_x:
        print(f"\nX-axis (width={width}):")
        print(f"  7-interval: spacing={target_7_x:.1f}px, magnitude={results_x['combined_7']:.3f}")
        print(f"  5-interval: spacing={target_5_x:.1f}px, magnitude={results_x['combined_5']:.3f}")
        print(f"  → WINNER: {results_x['winner']} (ratio: {results_x['ratio']:.2f}x)")

    if results_y:
        print(f"\nY-axis (height={height}):")
        print(f"  7-interval: spacing={target_7_y:.1f}px, magnitude={results_y['combined_7']:.3f}")
        print(f"  5-interval: spacing={target_5_y:.1f}px, magnitude={results_y['combined_5']:.3f}")
        print(f"  → WINNER: {results_y['winner']} (ratio: {results_y['ratio']:.2f}x)")

    print("\n" + "=" * 70)
    print("FINAL DECISION:")
    print("=" * 70)
    if results_x and results_y:
        if results_x['winner'] == results_y['winner']:
            print(f"✓ CONSISTENT: Both axes indicate {results_x['winner']}")
        else:
            print(f"✗ MIXED: X={results_x['winner']}, Y={results_y['winner']}")
    print("=" * 70)


if __name__ == "__main__":
    main()