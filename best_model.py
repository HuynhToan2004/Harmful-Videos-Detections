import cv2
import os
from moviepy.editor import VideoFileClip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import whisper
from model import MultimodalModel
from utils import VideoDataset, ToTensorNormalize
from train_and_eval import train_model, evaluate_model

device = 'cuda'
whisper_model = whisper.load_model("base",device=device)  
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

root_dir = '/data/npl/ICEK/News/DL/data/data/data_video/train' 

dataset = VideoDataset(root_dir=root_dir, transform=transform, tokenizer=tokenizer, text_model=text_model)
train = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  


dataset = VideoDataset(root_dir='/data/npl/ICEK/News/DL/data/data/data_video/val', transform=transform, tokenizer=tokenizer, text_model=text_model)
val = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  

num_epochs = 10
train_model(model, train, criterion, optimizer, num_epochs=num_epochs)
torch.save(model.state_dict(), '/data/npl/ICEK/News/DL/src/multimodal_model_less_complex.pth')

evaluate_model(model,val,device)





