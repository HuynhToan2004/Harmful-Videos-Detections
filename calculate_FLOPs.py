from ptflops import get_model_complexity_info
import torch
import torch.nn as nn
import cv2
import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import whisper
from model import MultimodalModel
from utils import VideoDataset, ToTensorNormalize
from train_and_eval import train_model, evaluate_model

# Thiết lập thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tải lại mô hình và tokenizer
whisper_model = whisper.load_model("base", device=device)
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model.eval()
text_model.to(device)

num_classes = len(os.listdir('/data/npl/ICEK/News/DL/data/data/data_video/train'))
from thop import profile

# Khởi tạo mô hình
model = MultimodalModel(
    num_classes=num_classes, 
    visual_hidden_size=256, 
    audio_hidden_size=128, 
    text_hidden_size=128, 
    embed_dim=256
)
model = model.to(device)
model.eval()

visual_input = torch.randn(1, 30, 3, 224, 224).to(device)
audio_input = torch.randn(1, 40, 40).to(device)
text_input = torch.randn(1, 768).to(device)

macs, params = profile(model, inputs=(visual_input, audio_input, text_input))
print(f"Total FLOPs: {macs / 1e9:.2f} GFLOPs")
print(f"Total Parameters: {params / 1e6:.2f} M")
