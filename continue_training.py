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

# Thiết lập thiết bị
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Tải lại mô hình và tokenizer
whisper_model = whisper.load_model("base", device=device)
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model.eval()
text_model.to(device)

print("Bản ít phức tạp nhưng đạt hiệu suất cao nhất: ")

num_classes = len(os.listdir('/data/npl/ICEK/News/DL/data/data/data_video/train'))
model = MultimodalModel(
    num_classes=num_classes, 
    visual_hidden_size=256, 
    audio_hidden_size=128, 
    text_hidden_size=128, 
    embed_dim=256
)

model = model.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)


transform = ToTensorNormalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])

train_root_dir = '/data/npl/ICEK/News/DL/data/data/data_video/train'
val_root_dir = '/data/npl/ICEK/News/DL/data/data/data_video/val'

# Tạo DataLoader cho tập huấn luyện
train_dataset = VideoDataset(root_dir=train_root_dir, transform=transform, tokenizer=tokenizer, text_model=text_model)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=0)

# Tạo DataLoader cho tập validation
val_dataset = VideoDataset(root_dir=val_root_dir, transform=transform, tokenizer=tokenizer, text_model=text_model)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=0)

# Tải lại trọng số mô hình từ tệp .pth
checkpoint_path = '/data/npl/ICEK/News/DL/src/multimodal_model_less_complex.pth'
if os.path.exists(checkpoint_path):
    model.load_state_dict(torch.load(checkpoint_path))
    print(f"Đã tải trọng số mô hình từ {checkpoint_path}")
else:
    print(f"Không tìm thấy checkpoint tại {checkpoint_path}, bắt đầu huấn luyện từ đầu.")


additional_epochs = 5
num_epochs = additional_epochs

train_model(model, train_loader, criterion, optimizer, num_epochs=num_epochs)

torch.save(model.state_dict(), '/data/npl/ICEK/News/DL/src/multimodal_model_less_complex_continued.pth')
print(f"Đã lưu trọng số mô hình sau khi tiếp tục")

evaluate_model(model, val_loader, device)
