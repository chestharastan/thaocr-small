import torch
import sys
import os

# Add parent directory to path so we can import from src
sys.path.append(os.getcwd())

from src.model import OCRRecModel
from src.config import Config, MODEL_PRESETS

def print_checkpoint_info(checkpoint_path):
    print(f"\n📢 Analyzing Checkpoint: {checkpoint_path}\n")
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ Error: Checkpoint file not found at {checkpoint_path}")
        return

    try:
        # Load the checkpoint on CPU to be safe
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        
        # 1. Extract Config
        if 'config' in checkpoint:
            cfg = checkpoint['config']
            print("🏗️  Model Architecture:")
            print(f"   • Name:      {cfg.model.name}")
            print(f"   • Backbone:  {cfg.model.backbone.type}")
            print(f"   • Head:      {cfg.model.head.type}")
            print(f"   • Layers:    {cfg.model.head.num_layers}")
            print(f"   • Hidden Dim:{cfg.model.head.d_model}")
            print(f"   • Img Height:{cfg.model.target_h}px")
        else:
            print("⚠️  No config found in checkpoint. Inferring from weights...")

        # 2. Extract Training Stats
        if 'epoch' in checkpoint:
            print(f"\n📅 Training Status:")
            print(f"   • Epochs Trained: {checkpoint['epoch']}")
            
        if 'best_cer' in checkpoint:
            print(f"   • Best CER:       {checkpoint['best_cer']:.4f} (Character Error Rate)")

        # 3. Parameter Count & Inference when config is missing
        # Re-instantiate model to count params exactly
        
        state_dict = checkpoint['model_state_dict']
        vocab_size = 0
        d_model = 0
        num_layers = 0
        
        # Find the final classification layer weight
        for key in state_dict.keys():
            if "head.fc.weight" in key:
                vocab_size = state_dict[key].shape[0]
                d_model = state_dict[key].shape[1]
                break
        
        # Count layers if not in config
        if 'config' not in checkpoint:
             max_layer = 0
             for key in state_dict.keys():
                 if "head.encoder.layers." in key:
                     try:
                        # Extract layer number
                        layer_num = int(key.split("head.encoder.layers.")[1].split(".")[0])
                        max_layer = max(max_layer, layer_num)
                     except:
                        pass
             num_layers = max_layer + 1

        if vocab_size > 0:
            if 'config' in checkpoint:
                 model_cfg = cfg.model
            else:
                 # Reconstruct minimal config from weights
                 from src.config import ModelConfig
                 model_cfg = ModelConfig()
                 model_cfg.head.d_model = d_model
                 model_cfg.head.num_layers = num_layers 
                 print(f"\n📢 Config inferred from weights (approx):")
                 print(f"   • Layers: {num_layers}")
                 print(f"   • d_model: {d_model}")
                 
            try:
               model = OCRRecModel.from_config(model_cfg, vocab_size=vocab_size)
               params = model.count_params()
               print(f"\n🧠 Model Complexity:")
               print(f"   • Total Params:     {params['total']:,}")
               print(f"   • Trainable Params: {params['trainable']:,}")
               print(f"   • Model Size:       ~{params['total_M']:.2f} Million Parameters")
            except:
               print("\n⚠️  Could not roughly instantiate model to count parameters (architecture mismatch).")
        else:
            print("\n⚠️  Could not determine vocab size to count parameters.")

    except Exception as e:
        print(f"\n❌ Error reading checkpoint: {e}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", help="Path to .pt checkpoint file")
    args = parser.parse_args()
    
    print_checkpoint_info(args.checkpoint)
