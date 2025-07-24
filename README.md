based on: https://huggingface.co/lym00/comfyui_nunchaku_lora_patch/blob/main/patch_comfyui_nunchaku_lora.py

# Universal LoRA Final Layer `adaLN` Patcher

This script patches `.safetensors` LoRA models that are missing `adaLN` (Adaptive Layer Normalization) weights in their final layer. This is a common issue with some LoRA models that can cause compatibility problems with certain loaders, such as the Nunchaku LoRA loader in ComfyUI, which expect these weights to be present.

## The Problem

When a LoRA model is trained without `adaLN` weights for the final layer, loaders that strictly require them will fail to load the model, often with a key mismatch error. This script provides a simple solution by adding dummy (zero-filled) `adaLN` weights, allowing the model to be loaded without errors.

## How It Works

The script automatically scans the `lora/` directory for any `.safetensors` files. For each file found, it inspects it for `final_layer.linear` weights. If these weights exist but the corresponding `final_layer.adaLN_modulation_1` weights are missing, the script performs the following actions:

1.  **Reads the existing `final_layer.linear` tensors** to determine their shape.
2.  **Generates new zero-filled "dummy" tensors** for `adaLN_modulation_1` that match the shape of the linear layer tensors.
3.  **Adds these new dummy tensors** to the model's state dictionary under the correct keys.
4.  **Saves the modified state dictionary** to a new `.safetensors` file in the same directory, with a `_patched` suffix (e.g., `my_lora_patched.safetensors`).

The script is designed to be robust and checks for common key prefixes (e.g., `lora_unet_final_layer`, `final_layer`) to maximize compatibility. It will skip any files that already have the `_patched` suffix.

## Usage

### Prerequisites

Make sure you have Python installed, along with the `torch` and `safetensors` libraries. If you don't have them, you can install them using pip:

```sh
pip install torch safetensors
```

### Steps

1.  Place all the LoRA models you want to patch into the `lora/` directory.
2.  **Run the script** from your terminal in the project directory:
    ```sh
    python patch_comfyui_nunchaku_lora.py
    ```
3.  The script will automatically find, patch, and save the new files in the `lora/` directory.

### Example Output

```
$ python patch_comfyui_nunchaku_lora.py

🔄 Universal final_layer.adaLN LoRA patcher (.safetensors)
Looking for .safetensors files in the 'lora' directory...

Found 1 file(s) to process.

-----------------------------------------------------
Processing: lora\fal-Realism-Detailer-Kontext.safetensors
✅ Loaded 130 tensors.
✅ Patch applied using prefix 'lora_unet_final_layer'.
✅ Patched file saved to: lora\fal-Realism-Detailer-Kontext_patched.safetensors
✅ Verification successful: `adaLN` keys are present.
-----------------------------------------------------
🎉 Done. Patched 1 file(s).
```

**Note:** This script does not create functional `adaLN` layers; it only adds zero-filled placeholder weights to ensure compatibility. The patched model should produce the same visual output as the original.