import os
import sys

def inspect_onnx(path):
    print(f"Inspecting {path}...")
    try:
        import onnx
        model = onnx.load(path)
        print("Model loaded successfully.")
        
        # approximate param count
        count = 0
        for tensor in model.graph.initializer:
            # product of dimensions
            dims = tensor.dims
            params = 1
            for d in dims:
                params *= d
            count += params
            
        print(f"Estimated Parameters: {count:,}")
        print(f"Estimated Size (float32): {count * 4 / (1024*1024):.2f} MB")
        
        # metadata
        for prop in model.metadata_props:
            print(f"Metadata: {prop.key} = {prop.value}")
            
    except ImportError:
        print("ONNX module not found. Cannot inspect structure.")
    except Exception as e:
        print(f"Error inspecting ONNX: {e}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        inspect_onnx(sys.argv[1])
