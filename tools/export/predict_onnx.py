"""
ThaoOCR — ONNX Inference Script (Padding Version)
Fixes the Transformer Reshape error by padding images to a fixed width.
"""
import os
import sys
import argparse
import json
import numpy as np
import cv2
import onnxruntime as ort

def preprocess_with_padding(image_path, target_h=32, target_w=256):
    """Load image, resize to height, then PAD to target_w to avoid Reshape errors."""
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")
    
    # 1. Resize height keeping aspect ratio
    h, w = img.shape[:2]
    scale = target_h / float(h)
    new_w = int(round(w * scale))
    
    # If the image is wider than target_w, we MUST resize it anyway (accuracy will drop)
    # If narrower, we pad.
    if new_w > target_w:
        img = cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_CUBIC)
        current_w = target_w
    else:
        img = cv2.resize(img, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        current_w = new_w

    # 2. Pad to exactly target_w (using white background = 255)
    # Most OCR models expect text on white or black background.
    # We'll use the border color of the original image (usually white for documents)
    pad_val = 255 
    padded = np.ones((target_h, target_w), dtype=np.uint8) * pad_val
    padded[:, :current_w] = img
    
    # 3. Normalize to [0, 1] then [-1, 1]
    x = padded.astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    
    # 4. Add batch/channel dims: [1, 1, H, W]
    x = x[np.newaxis, np.newaxis, :, :]
    return x

def ctc_decode_greedy(logits, itos, blank_id=0):
    """Greedy decoding for CTC logits."""
    # logits shape: [T, 1, V]
    preds = np.argmax(logits, axis=-1)
    if len(preds.shape) == 2:
        preds = preds[:, 0]
    
    chars = []
    prev = -1
    for p in preds:
        if p != blank_id and p != prev:
            if p < len(itos):
                chars.append(itos[p])
        prev = p
    return "".join(chars)

def main():
    parser = argparse.ArgumentParser(description="ThaoOCR ONNX Inference")
    parser.add_argument("--model", type=str, default="model.onnx")
    parser.add_argument("--vocab", type=str, default="model_vocab.json")
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--height", type=int, default=32)
    parser.add_argument("--width", type=int, default=1024, help="Must match width used during export")
    args = parser.parse_args()

    # 1. Load Vocab
    with open(args.vocab, "r", encoding="utf-8") as f:
        v = json.load(f)
        itos, blank_id = v["itos"], v.get("blank_id", 0)

    # 2. Preprocess with Padding
    img_tensor = preprocess_with_padding(args.image, target_h=args.height, target_w=args.width)

    # 3. Session
    session = ort.InferenceSession(args.model, providers=['CPUExecutionProvider'])
    
    # 4. Run
    inputs = {session.get_inputs()[0].name: img_tensor}
    logits = session.run(None, inputs)[0]

    # 5. Decode
    text = ctc_decode_greedy(logits, itos, blank_id)
    print(f"\nResult: {text}")

if __name__ == "__main__":
    main()
