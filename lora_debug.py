import torch
import os

# Load your style LoRA checkpoint
lora_path = "./outputs/train/train_2026-01-23T14-56-49/checkpoint-150/style/lora"
output_file = "lora_dimension_analysis.txt"

with open(output_file, 'w') as f:
    # Check for .pt or .safetensors files
    for file in os.listdir(lora_path):
        if file.endswith('.pt') or file.endswith('.bin') or file.endswith('.safetensors'):
            full_path = os.path.join(lora_path, file)
            
            if file.endswith('.safetensors'):
                from safetensors.torch import load_file
                state_dict = load_file(full_path)
            else:
                state_dict = torch.load(full_path, map_location='cpu')
            
            f.write(f"\n{'='*60}\n")
            f.write(f"File: {file}\n")
            f.write(f"{'='*60}\n\n")
            
            # Group by resolution
            dims_320 = []
            dims_640 = []
            dims_1280 = []
            other_dims = []
            
            for key, tensor in state_dict.items():
                if 'lora_down' in key and 'weight' in key:
                    in_features = tensor.shape[1]  # shape is (rank, in_features)
                    
                    if in_features == 320:
                        dims_320.append((key, tensor.shape))
                    elif in_features == 640:
                        dims_640.append((key, tensor.shape))
                    elif in_features == 1280:
                        dims_1280.append((key, tensor.shape))
                    else:
                        other_dims.append((key, tensor.shape, in_features))
            
            # Write 320-dim layers
            f.write(f"--- 320-dim LoRAs ({len(dims_320)} layers) ---\n")
            for key, shape in dims_320[:5]:  # First 5
                f.write(f"  {key}: {shape}\n")
            if len(dims_320) > 5:
                f.write(f"  ... and {len(dims_320) - 5} more\n")
            f.write("\n")
            
            # Write 640-dim layers
            f.write(f"--- 640-dim LoRAs ({len(dims_640)} layers) ---\n")
            for key, shape in dims_640[:5]:  # First 5
                f.write(f"  {key}: {shape}\n")
            if len(dims_640) > 5:
                f.write(f"  ... and {len(dims_640) - 5} more\n")
            f.write("\n")
            
            # Write 1280-dim layers
            f.write(f"--- 1280-dim LoRAs ({len(dims_1280)} layers) ---\n")
            for key, shape in dims_1280[:5]:  # First 5
                f.write(f"  {key}: {shape}\n")
            if len(dims_1280) > 5:
                f.write(f"  ... and {len(dims_1280) - 5} more\n")
            f.write("\n")
            
            # Write other dims if any
            if other_dims:
                f.write(f"--- Other dimensions ({len(other_dims)} layers) ---\n")
                for key, shape, dim in other_dims:
                    f.write(f"  {key}: {shape} (in_features={dim})\n")
                f.write("\n")
            
            # Summary
            f.write(f"\n{'='*60}\n")
            f.write(f"SUMMARY\n")
            f.write(f"{'='*60}\n")
            f.write(f"320-dim LoRAs:  {len(dims_320)}\n")
            f.write(f"640-dim LoRAs:  {len(dims_640)}\n")
            f.write(f"1280-dim LoRAs: {len(dims_1280)}\n")
            f.write(f"Other dims:     {len(other_dims)}\n")
            f.write(f"Total LoRA layers: {len(dims_320) + len(dims_640) + len(dims_1280) + len(other_dims)}\n")

print(f"Analysis saved to {output_file}")
