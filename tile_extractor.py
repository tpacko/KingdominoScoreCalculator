#!/usr/bin/env python3
"""
kingdomino_extract_tiles.py – 7×7 grid extractor, equal-size tiles.
"""

import cv2 as cv
import numpy as np
import sys
from pathlib import Path

def crop_to_grid(img, grid_size=7):
    H, W = img.shape[:2]
    h = H // grid_size * grid_size
    w = W // grid_size * grid_size
    return img[:h, :w]

def crop_tiles(board_img, grid_size=7):
    H, W = board_img.shape[:2]
    sH, sW = H // grid_size, W // grid_size
    tiles = []
    for r in range(grid_size):
        for c in range(grid_size):
            tile = board_img[r*sH:(r+1)*sH, c*sW:(c+1)*sW]
            tiles.append(tile)
    return tiles

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: python kingdomino_extract_tiles.py board.jpg")
        sys.exit(1)
    board = cv.imread(sys.argv[1])
    if board is None:
        print("cannot read", sys.argv[1])
        sys.exit(1)

    grid_size = 7
    board_cropped = crop_to_grid(board, grid_size)

    # Draw division lines for visualization
    board_lines = board_cropped.copy()
    H, W = board_cropped.shape[:2]
    sH, sW = H // grid_size, W // grid_size
    color = (0, 0, 255)  # Red

    for i in range(1, grid_size):
        y = i * sH
        x = i * sW
        cv.line(board_lines, (0, y), (W, y), color, 2)  # horizontal
        cv.line(board_lines, (x, 0), (x, H), color, 2)  # vertical

    cv.imshow("Division Lines", board_lines)
    print("Press any key in the image window to continue...")
    cv.waitKey(0)
    cv.destroyAllWindows()

    tiles_dir = Path("tiles")
    tiles_dir.mkdir(exist_ok=True)

    tiles = crop_tiles(board_cropped, grid_size)

    for idx, tile in enumerate(tiles, 1):
        out_path = tiles_dir / f"{idx}.png"
        cv.imwrite(str(out_path), tile)
    print(f"Saved {len(tiles)} tiles to {tiles_dir.resolve()}")
