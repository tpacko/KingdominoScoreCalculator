#!/usr/bin/env python3
"""
Manual Quadrilateral Selection and Warping (Aspect Ratio Edition)
================================================================
Click four points, warp and view,
Keeps aspect right, max 1000 too!

Usage:
  python manual_warp.py --image input.jpg --out warped.png
"""

import argparse
import cv2 as cv
import numpy as np
from pathlib import Path
import os

MAX_SIDE = 1000  # Max output side (px)

def order_quad(pts):
    pts = np.array(pts, dtype=np.float32)
    s = pts.sum(axis=1)
    diff = np.diff(pts, axis=1)
    tl = pts[np.argmin(s)]
    br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]
    bl = pts[np.argmax(diff)]
    return np.array([tl, tr, br, bl], dtype=np.float32)

def main():
    parser = argparse.ArgumentParser(description="Manual quad select and warp (aspect ratio, max 1000 px)")
    parser.add_argument("--image", required=True, help="Input image")
    parser.add_argument("--out", required=True, help="Warped output image")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise FileNotFoundError(img_path)
    bgr = cv.imread(str(img_path))
    if bgr is None:
        raise RuntimeError(f"Could not load image: {img_path}")

    # Display window scale for ease of use
    win_name = "Select 4 corners (click in any order, ESC when done)"
    h0, w0 = bgr.shape[:2]
    scale = min(1.0, 800.0 / max(h0, w0))
    disp_img = bgr if scale == 1.0 else cv.resize(bgr, (int(w0 * scale), int(h0 * scale)), interpolation=cv.INTER_AREA)
    clone = disp_img.copy()
    points = []

    def click_event(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN and len(points) < 4:
            x0 = int(x / scale)
            y0 = int(y / scale)
            points.append([x0, y0])
            cv.circle(clone, (x, y), 8, (0, 255, 0), -1)
            cv.imshow(win_name, clone)

    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, clone)
    cv.setMouseCallback(win_name, click_event)

    print("Click 4 corners of the board in any order. Press ESC when done.")
    while True:
        cv.imshow(win_name, clone)
        k = cv.waitKey(1)
        if len(points) == 4:
            break
        if k == 27:
            break
    cv.destroyAllWindows()

    if len(points) != 4:
        print("Not enough points selected, quitting—no pixels warping, no rects morphing.")
        return

    quad = order_quad(points)

    # Compute destination size based on input quad aspect ratio
    widthA = np.linalg.norm(quad[1] - quad[0])
    widthB = np.linalg.norm(quad[2] - quad[3])
    maxWidth = int(round(max(widthA, widthB)))
    heightA = np.linalg.norm(quad[3] - quad[0])
    heightB = np.linalg.norm(quad[2] - quad[1])
    maxHeight = int(round(max(heightA, heightB)))

    aspect = maxWidth / maxHeight if maxHeight > 0 else 1.0

    # Scale to fit MAX_SIDE
    scale_factor = min(MAX_SIDE / max(maxWidth, maxHeight), 1.0)
    out_w = int(round(maxWidth * scale_factor))
    out_h = int(round(maxHeight * scale_factor))

    dst = np.array([[0, 0], [out_w - 1, 0], [out_w - 1, out_h - 1], [0, out_h - 1]], dtype=np.float32)
    M = cv.getPerspectiveTransform(quad, dst)
    warped = cv.warpPerspective(bgr, M, (out_w, out_h))

    # Show the warped result
    win_out = "Warped result"
    disp_out = warped
    if max(out_w, out_h) > 800:
        disp_out = cv.resize(warped, (int(out_w * 800 / max(out_w, out_h)), int(out_h * 800 / max(out_w, out_h))), interpolation=cv.INTER_AREA)
    cv.imshow(win_out, disp_out)
    print("Showing warped result – press any key to close.")
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Ensure output has valid extension
    out_path = str(args.out)
    root, ext = os.path.splitext(out_path)
    if ext.lower() not in [".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"]:
        out_path = root + ".png"
    cv.imwrite(out_path, warped)
    print(f"[✓] Warped image saved to {out_path} – aspect fit, max side legit!")

if __name__ == "__main__":
    main()
