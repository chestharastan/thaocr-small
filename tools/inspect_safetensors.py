from safetensors.torch import load_file
import sys
import os

def inspect_safetensors(path):
    print(f"\n🔍 Inspecting: {path}\n")
    if not os.path.exists(path):
        print(f"❌ File not found: {path}")
        return

    try:
        tensors = load_file(path)
        print(f"📦 Total Tensors: {len(tensors)}")
        
        total_params = 0
        print(f"\n{'Tensor Name':<50} | {'Shape':<20} | {'Dtype'}")
        print("-" * 80)
        
        for name, tensor in tensors.items():
            shape_str = str(list(tensor.shape))
            dtype_str = str(tensor.dtype).replace("torch.", "")
            print(f"{name:<50} | {shape_str:<20} | {dtype_str}")
            total_params += tensor.numel()
            
        print("-" * 80)
        print(f"🧠 Total Parameters: {total_params:,}")
        print(f"💾 File Size: {os.path.getsize(path) / (1024*1024):.2f} MB")

    except Exception as e:
        print(f"❌ Error reading safetensors: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_safetensors(sys.argv[1])
    else:
        print("Usage: python tools/inspect_safetensors.py path/to/model.safetensors")
