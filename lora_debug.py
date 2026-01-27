import torch
import os

# Load your style LoRA checkpoint
lora_path = "./outputs/train/train_2026-01-23T14-56-49/checkpoint-150/style/lora"
output_file = "lora_dimension_analysis.txt"

with open(output_file, 'w') as f:
    # Check for .pt or .safetensors files
    for file in os.listdir(lora_path):
        if file.endswith('.pt') or file.endswith('.bin'):
            full_path = os.path.join(lora_path, file)
            weights = torch.load(full_path, map_location='cpu')
            
            f.write(f"\n{'='*60}\n")
            f.write(f"File: {file}\n")
            f.write(f"{'='*60}\n\n")
            
            # Check if it's a list or dict
            if isinstance(weights, list):
                f.write(f"Format: List of tensors\n")
                f.write(f"Total tensors: {len(weights)}\n\n")
                
                # Group by resolution
                dims_320 = []
                dims_640 = []
                dims_1280 = []
                other_dims = []
                
                # Weights are saved as: [up0, down0, up1, down1, ...]
                # lora_down tensors have shape (rank, in_features)
                # lora_up tensors have shape (out_features, rank)
                
                for i in range(0, len(weights), 2):
                    if i + 1 < len(weights):
                        up_weight = weights[i]
                        down_weight = weights[i + 1]
                        
                        # down_weight shape: (rank, in_features)
                        in_features = down_weight.shape[1]
                        out_features = up_weight.shape[0]
                        
                        layer_info = f"Layer {i//2}: down={down_weight.shape}, up={up_weight.shape}"
                        
                        if in_features == 320:
                            dims_320.append((i//2, layer_info, down_weight.shape, up_weight.shape))
                        elif in_features == 640:
                            dims_640.append((i//2, layer_info, down_weight.shape, up_weight.shape))
                        elif in_features == 1280:
                            dims_1280.append((i//2, layer_info, down_weight.shape, up_weight.shape))
                        else:
                            other_dims.append((i//2, layer_info, down_weight.shape, up_weight.shape, in_features))
                
                # Write 320-dim layers
                f.write(f"--- 320-dim LoRAs ({len(dims_320)} layers) ---\n")
                for idx, info, down_shape, up_shape in dims_320[:5]:
                    f.write(f"  {info}\n")
                if len(dims_320) > 5:
                    f.write(f"  ... and {len(dims_320) - 5} more\n")
                f.write("\n")
                
                # Write 640-dim layers
                f.write(f"--- 640-dim LoRAs ({len(dims_640)} layers) ---\n")
                for idx, info, down_shape, up_shape in dims_640[:5]:
                    f.write(f"  {info}\n")
                if len(dims_640) > 5:
                    f.write(f"  ... and {len(dims_640) - 5} more\n")
                f.write("\n")
                
                # Write 1280-dim layers
                f.write(f"--- 1280-dim LoRAs ({len(dims_1280)} layers) ---\n")
                for idx, info, down_shape, up_shape in dims_1280[:5]:
                    f.write(f"  {info}\n")
                if len(dims_1280) > 5:
                    f.write(f"  ... and {len(dims_1280) - 5} more\n")
                f.write("\n")
                
                # Write other dims
                if other_dims:
                    f.write(f"--- Other dimensions ({len(other_dims)} layers) ---\n")
                    for idx, info, down_shape, up_shape, dim in other_dims:
                        f.write(f"  {info} (in_features={dim})\n")
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
                
            elif isinstance(weights, dict):
                f.write(f"Format: Dictionary\n")
                f.write(f"Keys: {list(weights.keys())[:10]}\n")
            else:
                f.write(f"Format: Unknown type {type(weights)}\n")

print(f"Analysis saved to {output_file}")
