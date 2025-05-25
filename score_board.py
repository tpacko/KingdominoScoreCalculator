import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
from collections import defaultdict, deque
import argparse

# Config
IMG_SIZE = 128  # Must match training
TILE_CLASSES = ['f', 'me', 'mi', 'w', 'wa', 'wh', 'c']
CODE2TERR = {"f": "forest", "me": "meadow", "mi": "mine",
             "w": "water", "wa": "wasteland", "wh": "wheat", "c": "castle"}
CROWN_CLASSES = [0, 1, 2, 3]


def classify_tile_img(tile_img, model):
    tile_img = cv2.cvtColor(tile_img, cv2.COLOR_BGR2RGB)  # Fix for color order!
    img = Image.fromarray(tile_img).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    tile_pred, crown_pred = model.predict(x)
    tile_idx = np.argmax(tile_pred)
    crown_idx = np.argmax(crown_pred)
    tile_code = TILE_CLASSES[tile_idx]
    crown_count = CROWN_CLASSES[crown_idx]
    return tile_code, crown_count


def split_board_into_tiles(board_img, board_size):
    h, w, _ = board_img.shape
    tile_h, tile_w = h // board_size, w // board_size
    tiles = []
    for i in range(board_size):
        row = []
        for j in range(board_size):
            tile = board_img[
                   i * tile_h:(i + 1) * tile_h,
                   j * tile_w:(j + 1) * tile_w
                   ]
            row.append(tile)
        tiles.append(row)
    return tiles


def annotate_board(board_img, results, board_size):
    display = board_img.copy()
    h, w, _ = display.shape
    tile_h, tile_w = h // board_size, w // board_size
    for i in range(board_size):
        for j in range(board_size):
            tile_code, crowns = results[i][j]
            label = f"{CODE2TERR[tile_code]}:{crowns}"
            x = j * tile_w + 5
            y = (i + 1) * tile_h - 10
            color = (0, 255, 0) if tile_code != 'c' else (255, 0, 0)
            cv2.putText(display, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            cv2.rectangle(display, (j * tile_w, i * tile_h), ((j + 1) * tile_w - 1, (i + 1) * tile_h - 1),
                          (255, 255, 255), 1)
    return display


def find_connected_areas(results, board_size):
    visited = [[False] * board_size for _ in range(board_size)]
    areas = defaultdict(list)
    for i in range(board_size):
        for j in range(board_size):
            if visited[i][j]: continue
            terr, crowns = results[i][j]
            if terr == 'c':  # Skip castle
                continue
            queue = deque([(i, j)])
            group = []
            while queue:
                x, y = queue.popleft()
                if not (0 <= x < board_size and 0 <= y < board_size): continue
                if visited[x][y]: continue
                t, c = results[x][y]
                if t != terr: continue
                visited[x][y] = True
                group.append((x, y, c))
                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    queue.append((x + dx, y + dy))
            if group:
                areas[(terr, len(areas))] = group
    return areas


def compute_score(areas):
    total = 0
    details = []
    for area_id, group in areas.items():
        terr = area_id[0]
        size = len(group)
        crowns = sum(c for _, _, c in group)
        score = size * crowns
        if score > 0:
            details.append((CODE2TERR[terr], size, crowns, score))
            total += score
    return total, details


def main(board_path, board_size):
    # Load model
    model = tf.keras.models.load_model('tile_classifier.h5')
    # Load board image (already cropped)
    board_img = cv2.imread(board_path)
    if board_img is None:
        print("Could not find board image at", board_path)
        return

    # 1. Split board into tiles
    tiles = split_board_into_tiles(board_img, board_size)

    # 2. Show each tile, wait for input; ESC exits script, any other key goes to next tile
    print(f"Showing {board_size * board_size} extracted tiles. Press any key for next, ESC to exit before scoring.")
    for i, row in enumerate(tiles):
        for j, tile in enumerate(row):
            disp_tile = cv2.resize(tile, (250, 250), interpolation=cv2.INTER_CUBIC)
            cv2.imshow(f"Tile ({i},{j})", disp_tile)
            key = cv2.waitKey(0)
            cv2.destroyWindow(f"Tile ({i},{j})")
            if key == 27:  # ESC key
                print("ESC pressed. Exiting before classification/scoring.")
                cv2.destroyAllWindows()
                break

    # 3. Classify all tiles
    results = []
    for row in tiles:
        row_res = []
        for tile in row:
            tile_code, crown_count = classify_tile_img(tile, model)
            row_res.append((tile_code, crown_count))
        results.append(row_res)

    # 4. Show classification image
    classified_img = annotate_board(board_img, results, board_size)
    cv2.imshow("Classified Board", classified_img)
    cv2.imwrite("classified_board.png", classified_img)

    # 5. Find connected areas and compute scores
    areas = find_connected_areas(results, board_size)
    total, details = compute_score(areas)
    print("Area Scores:")
    for terr, size, crowns, score in details:
        print(f"{terr}: area={size}, crowns={crowns}, score={score}")
    print("TOTAL SCORE:", total)

    # 6. Optionally show connected areas (with colors)
    area_img = board_img.copy()
    for idx, ((terr, area_id), group) in enumerate(areas.items()):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        for (i, j, _) in group:
            h, w, _ = area_img.shape
            tile_h, tile_w = h // board_size, w // board_size
            cv2.rectangle(area_img,
                          (j * tile_w, i * tile_h),
                          ((j + 1) * tile_w - 1, (i + 1) * tile_h - 1),
                          color, 3)
    cv2.imshow("Connected Areas", area_img)
    cv2.imwrite("connected_areas.png", area_img)

    print("Press any key to exit windows...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Kingdomino board scorer")
    parser.add_argument('board_image', type=str, help='Path to cropped board image (e.g., board.png)')
    parser.add_argument('--board_size', type=int, default=7, help='Board size (default: 7 for 7x7)')
    args = parser.parse_args()
    main(args.board_image, args.board_size)
