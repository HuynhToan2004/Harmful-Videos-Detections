import cv2
import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet50_Weights
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import whisper
from model import MultimodalModel
from utils import VideoDataset, ToTensorNormalize
from train_and_eval import train_model, evaluate_model
from sklearn.metrics import classification_report

device = 'cuda'
whisper_model = whisper.load_model("base",device=device)  
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model.eval() 
text_model.to(device)

print("Bản ít phức tạp nhưng đạt hiệu suất cao nhất: ")

# Khởi tạo mô hình, loss function và optimizer
num_classes = len(os.listdir('/data/npl/ICEK/News/DL/data/data/data_video/train')) 

model = MultimodalModel(
    num_classes=num_classes, 
    visual_hidden_size=256, 
    audio_hidden_size=128, 
    text_hidden_size=128, 
    embed_dim=256
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Tải trọng số từ file .pth
checkpoint_path = "/data/npl/ICEK/News/DL/src/multimodal_model_less_complex_continued.pth"
model.load_state_dict(torch.load(checkpoint_path, map_location=device))
model.eval()
transform = ToTensorNormalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])

dataset = VideoDataset(root_dir='/data/npl/ICEK/News/DL/data/data/data_video/test_final', transform=transform, tokenizer=tokenizer, text_model=text_model)
val = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  

accuracy, predicted, labels = evaluate_model(model, val, device)

print("Classification Report của test_final:")
print(classification_report(
    labels,
    predicted,
    target_names=["pornographic", "normal", "offensive", "horrible", "violent"]
))