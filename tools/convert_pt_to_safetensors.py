import torch
import os
import sys

try:
    from safetensors.torch import save_file
except ImportError:
    print("❌ 'safetensors' module not found. Please install it with: pip install safetensors")
    sys.exit(1)

def convert_checkpoint(pt_path, out_path):
    print(f"Converting {pt_path} -> {out_path}...")
    
    if not os.path.exists(pt_path):
        print(f"❌ Source checkpoint not found: {pt_path}")
        return

    try:
        # Load PyTorch checkpoint
        checkpoint = torch.load(pt_path, map_location="cpu")
        
        # Extract state_dict
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        else:
            state_dict = checkpoint  # assume it's just the state_dict
            
        # Save as safetensors
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        save_file(state_dict, out_path)
        print(f"✅ Saved: {out_path}")
        
    except Exception as e:
        print(f"❌ Conversion failed: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="Input .pt checkpoint")
    parser.add_argument("--out", required=True, help="Output .safetensors file")
    args = parser.parse_args()
    
    convert_checkpoint(args.pt, args.out)
