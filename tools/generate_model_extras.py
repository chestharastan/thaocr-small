import json
import os
import yaml

def create_extras(model_dir):
    print(f"Processing {model_dir}...")
    
    # 1. Read Vocab
    vocab_path = os.path.join(model_dir, "model_vocab.json")
    if not os.path.exists(vocab_path):
        print(f"❌ {vocab_path} not found.")
        return

    with open(vocab_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        itos = data.get("itos", [])

    # 2. Write khmer_dict.txt (Skip BLANK=0, PAD=1)
    # Assuming standard ThaoOCR vocab structure
    dict_path = os.path.join(model_dir, "khmer_dict.txt")
    with open(dict_path, "w", encoding="utf-8") as f:
        for ch in itos[2:]: # Skip [BLANK] and [PAD]
            f.write(ch + "\n")
    print(f"✅ Created {dict_path}")

    # 3. Write config.yml (Small Preset)
    config_path = os.path.join(model_dir, "config.yml")
    config_data = {
        "model": {
            "name": "small",
            "target_h": 32,
            "backbone": {"type": "lightweight"},
            "head": {
                "type": "transformer_ctc",
                "d_model": 128,
                "num_layers": 2
            }
        },
        "vocab": {
            # Standard vocab config, but we have the specific list in json
            "blank_token": "[BLANK]",
            "pad_token": "[PAD]"
        }
    }
    
    with open(config_path, "w", encoding="utf-8") as f:
        yaml.dump(config_data, f, sort_keys=False, allow_unicode=True)
    print(f"✅ Created {config_path}")

if __name__ == "__main__":
    create_extras("model9k")
    create_extras("model90k")
