# patch_comfyui_nunchaku_lora

> **Deprecated** — This issue has been fixed in the latest Nunchaku release. See [nunchaku-tech/ComfyUI-nunchaku#406](https://github.com/nunchaku-tech/ComfyUI-nunchaku/issues/406).

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-yellow)](LICENSE)

A single-file Python utility that patches `.safetensors` LoRA models missing `adaLN` (Adaptive Layer Normalization) weights in their final layer — restoring compatibility with the Nunchaku LoRA loader in ComfyUI.

*Based on: [lym00/comfyui_nunchaku_lora_patch](https://huggingface.co/lym00/comfyui_nunchaku_lora_patch)*

## The Problem

Some LoRA models are trained without `final_layer.adaLN_modulation_1` weights. The Nunchaku loader strictly requires these keys and fails with a key mismatch error when they are absent.

## How It Works

1. Scans `lora/` directory for all `.safetensors` files (skips `_patched` files)
2. For each file: detects `final_layer.linear` weights and checks for missing `adaLN_modulation_1` counterparts
3. Generates zero-filled dummy tensors matching the shape of the linear weights
4. Saves a new `_patched.safetensors` file — original is never modified

> The patched model produces **identical visual output** to the original. The dummy weights are zero-filled placeholders purely for loader compatibility — they have no functional effect.

## Tech Stack

| Dependency | Purpose |
|---|---|
| Python 3.8+ | Runtime |
| `torch` | Tensor creation and shape introspection |
| `safetensors` | Read / write `.safetensors` model files |

## Usage

```bash
pip install torch safetensors
```

1. Place LoRA models to patch in the `lora/` directory
2. Run the script:
   ```bash
   python patch_comfyui_nunchaku_lora.py
   ```
3. Patched files are saved as `lora/<original_name>_patched.safetensors`

**Example output:**
```
🔄 Universal final_layer.adaLN LoRA patcher (.safetensors)
Found 1 file(s) to process.

Processing: lora\fal-Realism-Detailer-Kontext.safetensors
✅ Loaded 130 tensors.
✅ Patch applied using prefix 'lora_unet_final_layer'.
✅ Patched file saved to: lora\fal-Realism-Detailer-Kontext_patched.safetensors
✅ Verification successful: adaLN keys are present.
🎉 Done. Patched 1 file(s).
```

## License

MIT
