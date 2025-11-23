import torch
import torch.nn as nn
from torch.utils.mobile_optimizer import optimize_for_mobile

from model import BoardKeypointNet


def main():
    INPUT_MODEL_PATH = "checkpoints/best.pt"
    OUTPUT_MODEL_NAME = "board_keypoint_detector.ptl"
    INPUT_IMAGE_SIZE = 512

    print(f"--- Starting Conversion for {INPUT_MODEL_PATH} ---")

    # 1. Instantiate the model
    model = BoardKeypointNet()

    # 2. Load the trained weights
    try:
        # Load raw file
        state_dict = torch.load(INPUT_MODEL_PATH, map_location='cpu')

        # Unwrap 'state_dict' key if it exists (common in PyTorch Lightning/standard saves)
        if 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']

        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[7:]  # Remove 'module.'
            new_state_dict[name] = v

        # 3. Load into model with STRICT=TRUE
        # This will CRASH if the weights are wrong, rather than failing silently.
        model.load_state_dict(new_state_dict, strict=True)
        print("SUCCESS: Weights loaded with strict matching!")

    except RuntimeError as e:
        print("\n!!! CRITICAL ERROR: WEIGHT MISMATCH !!!")
        print("The keys in your .pt file do not match the model architecture.")
        print(f"Error details: {e}")
        return
    except FileNotFoundError:
        print(f"ERROR: Could not find {INPUT_MODEL_PATH}")
        return

    # 4. Switch to Evaluation Mode
    model.eval()

    # 5. Trace
    print("Tracing model...")
    example_input = torch.rand(1, 3, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE)

    try:
        traced_script_module = torch.jit.trace(model, example_input)
    except Exception as e:
        print(f"ERROR during tracing: {e}")
        return

    # 6. Optimize
    print("Optimizing for mobile...")
    optimized_traced_model = optimize_for_mobile(traced_script_module)

    # 7. Save
    # optimization breaks the model for some reason
    # optimized_traced_model._save_for_lite_interpreter(OUTPUT_MODEL_NAME)
    traced_script_module._save_for_lite_interpreter(OUTPUT_MODEL_NAME)
    print(f"DONE. Saved to {OUTPUT_MODEL_NAME}")


if __name__ == "__main__":
    main()