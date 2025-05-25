import os
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image

# Config
IMG_SIZE = 128  # Must match training
TILE_CLASSES = ['f', 'me', 'mi', 'w', 'wa', 'wh', 'c']
CODE2TERR = {"f":"forest","me":"meadow","mi":"mine",
             "w":"water","wa":"wasteland","wh":"wheat","c":"castle"}
CROWN_CLASSES = [0, 1, 2, 3]

def classify_tile(img_path, model):
    img = Image.open(img_path).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
    x = np.array(img) / 255.0
    x = np.expand_dims(x, axis=0)
    tile_pred, crown_pred = model.predict(x)
    tile_idx = np.argmax(tile_pred)
    crown_idx = np.argmax(crown_pred)
    tile_code = TILE_CLASSES[tile_idx]
    tile_name = CODE2TERR[tile_code]
    crown_count = CROWN_CLASSES[crown_idx]
    return tile_name, crown_count

def main():
    # Load model
    model = tf.keras.models.load_model('tile_classifier.h5')
    tiles_folder = 'tiles'
    for fname in os.listdir(tiles_folder):
        if not fname.endswith('.png'):
            continue
        path = os.path.join(tiles_folder, fname)
        # Classify
        tile_name, crown_count = classify_tile(path, model)
        # Load for display
        img_cv = cv2.imread(path)
        # Resize for visibility if needed
        img_cv = cv2.resize(img_cv, (250, 250), interpolation=cv2.INTER_CUBIC)
        # Title text
        label = f"{tile_name} | crowns: {crown_count}"
        # Put label text on image
        cv2.putText(img_cv, label, (5, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 0), 1)
        cv2.imshow('Tile Classification', img_cv)
        print(f"File: {fname}   =>   {label}")
        key = cv2.waitKey(0)  # Wait for key press to continue
        if key == 27:  # ESC key to break
            break
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
