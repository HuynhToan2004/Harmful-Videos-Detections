import os
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModel
import whisper
from utils import extract_frames, extract_audio_mfcc_and_text
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("large", device=device)
tokenizer = AutoTokenizer.from_pretrained("/assets/phobert-base")
text_model = AutoModel.from_pretrained("/assets/phobert-base").to(device).eval()

root_dir = 'data/train'
output_dir = '/Harmful-Videos-Detections/features/train'
os.makedirs(output_dir, exist_ok=True)

for cls in tqdm(os.listdir(root_dir)):
    cls_path = os.path.join(root_dir, cls)
    out_cls_path = os.path.join(output_dir, cls)
    os.makedirs(out_cls_path, exist_ok=True)

    for file in os.listdir(cls_path):
        if not file.endswith(('.mp4', '.avi', '.mov')):
            continue

        video_path = os.path.join(cls_path, file)
        save_path = os.path.join(out_cls_path, file.replace('.mp4', '.pt'))

        frames = extract_frames(video_path)  # (30, 224, 224, 3)
        frames = np.transpose(frames, (0, 3, 1, 2)) / 255.0
        frames = torch.tensor(frames, dtype=torch.float)

        mfcc, text = extract_audio_mfcc_and_text(video_path)
        mfcc = torch.tensor(mfcc, dtype=torch.float)

        if text != "":
            inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128).to(device)
            with torch.no_grad():
                text_output = text_model(**inputs)
                text_embedding = text_output.last_hidden_state[:, 0, :].squeeze(0).cpu()
        else:
            text_embedding = torch.zeros(text_model.config.hidden_size)

        torch.save({
            "frames": frames,
            "mfcc": mfcc,
            "text_embedding": text_embedding
        }, save_path)
