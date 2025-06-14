import tensorflow as tf
import argparse

def convert_h5_to_tflite(keras_model_path, tflite_model_path, quantize=False):
    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_model_path, "wb") as f:
        f.write(tflite_model)
    print(f"TFLite model saved to {tflite_model_path}")

def main():
    parser = argparse.ArgumentParser(description="Convert Keras .h5 model to TensorFlow Lite .tflite")
    parser.add_argument("--h5_path", help="Path to the input .h5 Keras model", default="./tile_classifier.h5")
    parser.add_argument("--tflite_path", help="Path to output .tflite file", default="./tile_classifier.tflite")
    parser.add_argument("--quantize", action="store_true", help="Enable quantization (smaller/faster models)")
    args = parser.parse_args()
    convert_h5_to_tflite(args.h5_path, args.tflite_path, args.quantize)

if __name__ == "__main__":
    main()
