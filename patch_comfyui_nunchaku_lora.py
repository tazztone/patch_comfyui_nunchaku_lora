import safetensors.torch
from safetensors import safe_open
import torch
import os
import glob


def patch_final_layer_adaLN(state_dict, prefix="lora_unet_final_layer", verbose=True):
    """
    Add dummy adaLN weights if missing, using final_layer_linear shapes as reference.
    Args:
        state_dict (dict): keys -> tensors
        prefix (str): base name for final_layer keys
        verbose (bool): print debug info
    Returns:
        dict: patched state_dict
    """
    final_layer_linear_down = None
    final_layer_linear_up = None

    adaLN_down_key = f"{prefix}_adaLN_modulation_1.lora_down.weight"
    adaLN_up_key = f"{prefix}_adaLN_modulation_1.lora_up.weight"
    linear_down_key = f"{prefix}_linear.lora_down.weight"
    linear_up_key = f"{prefix}_linear.lora_up.weight"

    if verbose:
        print(f"\n🔍 Checking for final_layer keys with prefix: '{prefix}'")
        print(f"   Linear down: {linear_down_key}")
        print(f"   Linear up:   {linear_up_key}")

    if linear_down_key in state_dict:
        final_layer_linear_down = state_dict[linear_down_key]
    if linear_up_key in state_dict:
        final_layer_linear_up = state_dict[linear_up_key]

    has_adaLN = adaLN_down_key in state_dict and adaLN_up_key in state_dict
    has_linear = final_layer_linear_down is not None and final_layer_linear_up is not None

    if verbose:
        print(f"   ✅ Has final_layer.linear: {has_linear}")
        print(f"   ✅ Has final_layer.adaLN_modulation_1: {has_adaLN}")

    if has_linear and not has_adaLN:
        dummy_down = torch.zeros_like(final_layer_linear_down)
        dummy_up = torch.zeros_like(final_layer_linear_up)
        state_dict[adaLN_down_key] = dummy_down
        state_dict[adaLN_up_key] = dummy_up

        if verbose:
            print(f"✅ Added dummy adaLN weights:")
            print(f"   {adaLN_down_key} (shape: {dummy_down.shape})")
            print(f"   {adaLN_up_key} (shape: {dummy_up.shape})")
    else:
        if verbose:
            print("✅ No patch needed — adaLN weights already present or no final_layer.linear found.")

    return state_dict


def main():
    print("🔄 Universal final_layer.adaLN LoRA patcher (.safetensors)")
    print("Looking for .safetensors files in the 'lora' directory...")

    lora_dir = "lora"
    lora_files = glob.glob(os.path.join(lora_dir, "*.safetensors"))

    if not lora_files:
        print(f"\n❌ No `.safetensors` files found in the '{lora_dir}' directory.")
        return

    print(f"\nFound {len(lora_files)} file(s) to process.")
    patched_count = 0

    for input_path in lora_files:
        if "_patched" in os.path.basename(input_path):
            print(f"\n⏭️  Skipping already patched file: {input_path}")
            continue

        print(f"\n-----------------------------------------------------")
        print(f"Processing: {input_path}")

        base, ext = os.path.splitext(input_path)
        output_path = f"{base}_patched{ext}"

        # Load
        state_dict = {}
        with safe_open(input_path, framework="pt", device="cpu") as f:
            for k in f.keys():
                state_dict[k] = f.get_tensor(k)
        
        print(f"✅ Loaded {len(state_dict)} tensors.")

        # Try common prefixes in order
        prefixes = [
            "lora_unet_final_layer",
            "final_layer",
            "base_model.model.final_layer"
        ]
        patched = False
        before_count = len(state_dict)

        for prefix in prefixes:
            state_dict = patch_final_layer_adaLN(state_dict, prefix=prefix, verbose=False)
            if len(state_dict) > before_count:
                print(f"✅ Patch applied using prefix '{prefix}'.")
                patched = True
                break  # Stop after the first successful patch

        if not patched:
            print("✅ No patch needed for this file.")
            continue

        # Save
        safetensors.torch.save_file(state_dict, output_path)
        print(f"✅ Patched file saved to: {output_path}")
        patched_count += 1

        # Verify
        with safe_open(output_path, framework="pt", device="cpu") as f:
            has_adaLN_after = any("adaLN_modulation_1" in k for k in f.keys())
            if has_adaLN_after:
                print("✅ Verification successful: `adaLN` keys are present.")
            else:
                print("❌ Verification failed: `adaLN` keys are missing in the output file.")

    print(f"\n-----------------------------------------------------")
    print(f"🎉 Done. Patched {patched_count} file(s).")


if __name__ == "__main__":
    main()