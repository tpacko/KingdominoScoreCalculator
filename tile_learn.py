import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import random
from collections import defaultdict
import cv2

# -------------------------------
# CONFIGURATION
# -------------------------------
IMG_SIZE = 128  # Change if needed
BATCH_SIZE = 64
EPOCHS = 30

CODE2TERR = {
    "f": "forest",
    "me": "meadow",
    "mi": "mine",
    "w": "water",
    "wa": "wasteland",
    "wh": "wheat",
    "c": "castle"
}

TILE_CLASSES = list(CODE2TERR.keys())
CROWN_CLASSES = [0, 1, 2, 3]

# -------------------------------
# DATA LOADING + AUGMENTATION
# -------------------------------

def parse_filename(filename):
    name = filename.split('.')[0]
    parts = name.split('_')
    tile = ''.join([c for c in parts[0] if not c.isdigit()])
    crown = 0
    if len(parts) > 1:
        for p in parts[1:]:
            if p.startswith('c'):
                crown = int(p[1:])
    return tile, crown

def load_tile_images(folder, min_images_per_class=1):
    data = defaultdict(list)
    for fname in os.listdir(folder):
        if fname.endswith('.png'):
            tile, crown = parse_filename(fname)
            if tile in TILE_CLASSES and crown in CROWN_CLASSES:
                img = Image.open(os.path.join(folder, fname)).convert('RGB').resize((IMG_SIZE, IMG_SIZE))
                data[(tile, crown)].append(np.array(img) / 255.0)

    # Print number of images per class before balancing
    print("Image counts per class (tile, crown) BEFORE upsampling:")
    for (tile, crown), imgs in sorted(data.items()):
        print(f"  ({tile}, {crown}): {len(imgs)} images")

    # Filter out classes with not enough images (optional)
    filtered_data = {k: v for k, v in data.items() if len(v) >= min_images_per_class}
    if not filtered_data:
        raise ValueError("No classes with enough images. Lower min_images_per_class or add more data.")

    # Upsample all classes to the max class size
    max_count = max(len(imgs) for imgs in filtered_data.values())
    print(f"\nUpsampling all classes to {max_count} images (the largest group).\n")

    balanced_images, tile_labels, crown_labels = [], [], []
    post_balance_counts = defaultdict(int)

    for (tile, crown), imgs in filtered_data.items():
        sampled_imgs = random.choices(imgs, k=max_count)  # Always upsample with replacement
        balanced_images.extend(sampled_imgs)
        tile_labels.extend([TILE_CLASSES.index(tile)] * max_count)
        crown_labels.extend([CROWN_CLASSES.index(crown)] * max_count)
        post_balance_counts[(tile, crown)] = max_count

    # Print number of images per class after upsampling
    print("Image counts per class (tile, crown) AFTER upsampling:")
    for (tile, crown), count in sorted(post_balance_counts.items()):
        print(f"  ({tile}, {crown}): {count} images")

    print(f"\nTotal dataset size after upsampling: {len(balanced_images)} images\n")

    return np.array(balanced_images), np.array(tile_labels), np.array(crown_labels)

def add_transparent_overlay(image):
    pil_img = Image.fromarray((image * 255).astype(np.uint8))
    overlay = Image.new('RGBA', pil_img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    color = (255, 255, 255, random.randint(20, 60)) if random.random() < 0.5 else (0, 0, 0, random.randint(20, 60))
    shape = [(random.randint(0, IMG_SIZE//2), random.randint(0, IMG_SIZE//2)),
             (random.randint(IMG_SIZE//2, IMG_SIZE), random.randint(IMG_SIZE//2, IMG_SIZE))]
    draw.rectangle(shape, fill=color)
    combined = Image.alpha_composite(pil_img.convert('RGBA'), overlay)
    return np.array(combined.convert('RGB')) / 255.0

def augment_image(img):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.image.rot90(img, k=1)
    img = tf.image.random_hue(img, 0.06)
    img = tf.image.random_saturation(img, 0.8, 1.2)
    img = tf.image.random_brightness(img, 0.2)
    img = tf.clip_by_value(img, 0, 1)
    img_np = img.numpy()
    return add_transparent_overlay(img_np)

def augment_batch(batch):
    return np.array([augment_image(img) for img in batch])

# -------------------------------
# MODEL DEFINITION
# -------------------------------

def build_model():
    inp = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.Conv2D(32, 3, activation='relu', padding='same')(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(256, 3, activation='relu', padding='same')(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Flatten()(x)
    x = layers.Dense(512, activation='sigmoid')(x)
    tile_type = layers.Dense(len(TILE_CLASSES), activation='softmax', name='tile_type')(x)
    crown_count = layers.Dense(len(CROWN_CLASSES), activation='softmax', name='crown_count')(x)
    model = models.Model(inputs=inp, outputs=[tile_type, crown_count])
    return model

# -------------------------------
# MAIN FUNCTION
# -------------------------------


def draw_label(img, text, pos=(5, 20)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.55
    color = (0, 0, 0)
    thickness = 2
    img = cv2.putText(img, text, pos, font, font_scale, (255,255,255), thickness+2, cv2.LINE_AA)
    img = cv2.putText(img, text, pos, font, font_scale, color, thickness, cv2.LINE_AA)
    return img

def main():
    images, tile_labels, crown_labels = load_tile_images('tiles', min_images_per_class=1)
    tile_labels_cat = to_categorical(tile_labels, num_classes=len(TILE_CLASSES))
    crown_labels_cat = to_categorical(crown_labels, num_classes=len(CROWN_CLASSES))

    X_train, X_val, y_tile_train, y_tile_val, y_crown_train, y_crown_val = train_test_split(
        images, tile_labels_cat, crown_labels_cat, test_size=0.05, random_state=42
    )

    # --- Interactive periodic preview of 9 random augmented images with labels (OpenCV) ---
    n_samples = len(X_train)
    indices = list(range(n_samples))
    batch_size = 9
    stop_preview = False

    while not stop_preview:
        rand_idxs = random.sample(indices, min(batch_size, n_samples))
        imgs = []
        for idx in rand_idxs:
            aug_img = augment_image(X_train[idx])
            tile_idx = np.argmax(y_tile_train[idx])
            crown_idx = np.argmax(y_crown_train[idx])
            tile_name = CODE2TERR[TILE_CLASSES[tile_idx]]
            label = f"{tile_name}, {CROWN_CLASSES[crown_idx]}c"
            # Prepare for cv2: float32->uint8, RGB->BGR
            img_disp = (aug_img * 255).astype(np.uint8)
            img_disp = cv2.cvtColor(img_disp, cv2.COLOR_RGB2BGR)
            img_disp = draw_label(img_disp, label)
            imgs.append(img_disp)
        # Compose 3x3 grid
        blank = np.ones_like(imgs[0]) * 200
        while len(imgs) < 9:
            imgs.append(blank.astype(np.uint8))
        row1 = np.concatenate(imgs[0:3], axis=1)
        row2 = np.concatenate(imgs[3:6], axis=1)
        row3 = np.concatenate(imgs[6:9], axis=1)
        grid = np.concatenate([row1, row2, row3], axis=0)
        cv2.imshow("9 Random Augmented Images (ENTER=next, ESC=continue to training)", grid)
        key = cv2.waitKey(0)
        if key == 27:  # ESC
            stop_preview = True
            cv2.destroyAllWindows()
        # Otherwise (ENTER, etc.), just continue the loop

    print("Continuing to learning...")

    # --------- Model Training ---------
    def train_gen():
        idxs = np.arange(len(X_train))
        while True:
            np.random.shuffle(idxs)
            for start in range(0, len(idxs), BATCH_SIZE):
                end = min(start + BATCH_SIZE, len(idxs))
                batch_idx = idxs[start:end]
                if random.random() < 0.5:
                    batch_x = augment_batch(X_train[batch_idx])
                else:
                    batch_x = X_train[batch_idx]
                batch_y_tile = y_tile_train[batch_idx]
                batch_y_crown = y_crown_train[batch_idx]
                yield batch_x, {'tile_type': batch_y_tile, 'crown_count': batch_y_crown}

    steps_per_epoch = int(np.ceil(len(X_train) / BATCH_SIZE))

    model = build_model()
    model.compile(
        optimizer='adam',
        loss={'tile_type': 'categorical_crossentropy', 'crown_count': 'categorical_crossentropy'},
        metrics={'tile_type': 'accuracy', 'crown_count': 'accuracy'}
    )

    history = model.fit(
        train_gen(),
        steps_per_epoch=steps_per_epoch,
        epochs=EPOCHS,
        validation_data=(X_val, {'tile_type': y_tile_val, 'crown_count': y_crown_val}),
        shuffle=True
    )
    model.save('tile_classifier.h5')
    print('Training done and model saved!')

if __name__ == "__main__":
    main()
