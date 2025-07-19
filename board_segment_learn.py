import os
import glob
import re
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
import cv2

# 1. Load image and mask, resize to 512x512
def load_image_pair(image_path, mask_path, img_size=(512, 512)):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_png(image, channels=3)
    image = tf.image.resize(image, img_size)
    image = tf.cast(image, tf.float32) / 255.0

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    mask = tf.image.resize(mask, img_size, method='nearest')
    mask = tf.cast(mask > 0, tf.float32)

    return image, mask

# 2. Dataset loader with logging
def create_dataset(image_dir, mask_dir, img_size=(512, 512), batch_size=8):
    all_image_paths = glob.glob(os.path.join(image_dir, '*'))

    valid_pairs = []
    for path in sorted(all_image_paths):
        filename = os.path.basename(path)
        match = re.fullmatch(r'game(\d+)\.[a-zA-Z]+', filename)  # exact match
        if match:
            number = match.group(1)
            mask_path = os.path.join(mask_dir, f"game{number}.png")
            if os.path.exists(mask_path):
                valid_pairs.append((path, mask_path))

    if not valid_pairs:
        raise ValueError("No valid image-mask pairs found.")

    print("📝 Using the following image-mask pairs:")
    for img, mask in valid_pairs:
        print(f"  📷 {img}  🖼️ {mask}")

    image_paths, mask_paths = zip(*valid_pairs)
    dataset = tf.data.Dataset.from_tensor_slices((list(image_paths), list(mask_paths)))
    dataset = dataset.map(lambda x, y: load_image_pair(x, y, img_size), num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(100).batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

# 3. U-Net model
def build_unet(input_shape=(512, 512, 3)):
    inputs = tf.keras.Input(shape=input_shape)
    c1 = layers.Conv2D(16, 3, activation='relu', padding='same')(inputs)
    p1 = layers.MaxPooling2D()(c1)
    c2 = layers.Conv2D(32, 3, activation='relu', padding='same')(p1)
    p2 = layers.MaxPooling2D()(c2)
    b = layers.Conv2D(64, 3, activation='relu', padding='same')(p2)
    u1 = layers.Conv2DTranspose(32, 3, strides=2, activation='relu', padding='same')(b)
    u2 = layers.Conv2DTranspose(16, 3, strides=2, activation='relu', padding='same')(u1)
    outputs = layers.Conv2D(1, 1, activation='sigmoid')(u2)
    return models.Model(inputs, outputs)

# 4. Training loop
def train_model(model, dataset, epochs=10):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(dataset, epochs=epochs)

# 5. Show predictions after training
def show_predictions(model, dataset, num=3):
    for images, masks in dataset.take(1):
        preds = model.predict(images)
        for i in range(min(num, images.shape[0])):
            fig, axs = plt.subplots(1, 3, figsize=(12, 4))
            axs[0].imshow(images[i])
            axs[0].set_title("Image")
            axs[1].imshow(masks[i, ..., 0], cmap='gray')
            axs[1].set_title("Ground Truth")
            axs[2].imshow(preds[i, ..., 0], cmap='gray')
            axs[2].set_title("Prediction")
            for ax in axs:
                ax.axis('off')
            plt.tight_layout()
            plt.show()

# 6. Optional preview before training
def preview_dataset(dataset, num=5):
    for images, masks in dataset.take(1):
        for i in range(min(num, images.shape[0])):
            img = images[i].numpy()
            msk = masks[i].numpy().squeeze()

            # Merge side-by-side
            img_disp = (img * 255).astype(np.uint8)
            msk_disp = (msk * 255).astype(np.uint8)
            msk_disp_rgb = cv2.cvtColor(msk_disp, cv2.COLOR_GRAY2RGB)
            combined = np.hstack([img_disp, msk_disp_rgb])

            cv2.imshow(f'Preview {i+1}', combined)
            key = cv2.waitKey(0)
            cv2.destroyAllWindows()
            if key == 27:  # ESC key
                print("⏩ Skipping preview.")
                return

# 7. Main
def main():
    dataset = create_dataset('files', 'masks', img_size=(512, 512), batch_size=4)
    preview_dataset(dataset)
    model = build_unet(input_shape=(512, 512, 3))
    train_model(model, dataset, epochs=100)
    show_predictions(model, dataset)

if __name__ == '__main__':
    main()
