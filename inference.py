import torch
import os
import numpy as np
from model import MultimodalModel

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)
num_classes = 6  # 

model = MultimodalModel(
    num_classes=num_classes,
    visual_hidden_size=256,
    audio_hidden_size=128,
    text_hidden_size=128,
    embed_dim=256
)
model.load_state_dict(torch.load(
    "/Harmful-Videos-Detections/multimodal_model_less_complex_v3.pth"
))
model.to(device)
model.eval()

def map_class_idx(pred_class: int):
    target_names = ["horrible", "normal", "offensive", "pornographic", "supertitious", "violent"]
    
    if 0 <= pred_class < len(target_names):
        return target_names[pred_class]


def infer_from_feature_file(feature_path):
    # Load data
    data = torch.load(feature_path)
    frames = data['frames'].unsqueeze(0).to(device)         #  [1, T, C, H, W] 
    mfcc = data['mfcc'].unsqueeze(0).to(device)                          # [1, D]
    text_embedding = data['text_embedding'].unsqueeze(0).to(device)     # [1, D]

    with torch.no_grad():
        output = model(frames, mfcc, text_embedding)
        probs = torch.softmax(output, dim=1)
        pred_class = torch.argmax(probs, dim=1).item()
    
    return map_class_idx(pred_class), probs.squeeze().cpu().numpy()


feature_file = "/Harmful-Videos-Detections/features/test/superstitious/0123.pt"  
pred_class, probs = infer_from_feature_file(feature_file)
print(f"Predicted class: {pred_class}")
print(f"Class probabilities: {probs}")
