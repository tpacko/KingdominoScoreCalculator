#!/usr/bin/env python3
"""
kingdomino_extract_tiles.py – 7×7 or 5x5 grid extractor, equal-size tiles.
"""

import cv2 as cv
import numpy as np
import sys
import random
from pathlib import Path

# -------------------------------
# CONFIGURATION
# -------------------------------
BOARD_FILES = [
    'boards/game19_board.jpg',
    'boards/game20_board.jpg',
    'boards/game21_board.jpg',
    'boards/game22_board.jpg',
]
GRID_SIZE = 5  # Set your grid size here (e.g., 7 or 5)

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
    tiles_dir = Path("tiles")
    tiles_dir.mkdir(exist_ok=True)

    total_tiles = 0
    for board_path in BOARD_FILES:
        board = cv.imread(board_path)
        if board is None:
            print("cannot read", board_path)
            continue

        board_cropped = crop_to_grid(board, GRID_SIZE)

        # Draw division lines for visualization
        board_lines = board_cropped.copy()
        H, W = board_cropped.shape[:2]
        sH, sW = H // GRID_SIZE, W // GRID_SIZE
        color = (0, 0, 255)  # Red

        for i in range(1, GRID_SIZE):
            y = i * sH
            x = i * sW
            cv.line(board_lines, (0, y), (W, y), color, 2)  # horizontal
            cv.line(board_lines, (x, 0), (x, H), color, 2)  # vertical

        cv.imshow(f"Division Lines: {board_path}", board_lines)
        print(f"Press any key in the image window to continue for {board_path}...")
        cv.waitKey(0)
        cv.destroyAllWindows()

        tiles = crop_tiles(board_cropped, GRID_SIZE)

        for tile in tiles:
            # Generate a unique random number for the filename (no prefix)
            while True:
                rand_num = random.randint(100000, 999999)
                out_path = tiles_dir / f"{rand_num}.png"
                if not out_path.exists():
                    break
            cv.imwrite(str(out_path), tile)
            total_tiles += 1
    print(f"Saved {total_tiles} tiles to {tiles_dir.resolve()} (randomized filenames)")
