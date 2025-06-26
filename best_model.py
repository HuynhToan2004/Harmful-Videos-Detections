import cv2
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/Harmful-Videos-Detections/ffmpeg/bin/ffmpeg"
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
from utils import VideoDataset, ToTensorNormalize, PreExtractedFeatureDataset
from train_and_eval import train_model, evaluate_model

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ",device)
whisper_model = whisper.load_model("large",device=device)  
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/assets/phobert-base")
text_model.eval() 
text_model.to(device)

print("Bản ít phức tạp nhưng đạt hiệu suất cao nhất: ")

num_classes = len(os.listdir('/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/home/anhkhoa/spark_video_streaming/data/train')) 

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

train_dataset = PreExtractedFeatureDataset('/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/Harmful-Videos-Detections/features/train')
val_dataset = PreExtractedFeatureDataset('/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/Harmful-Videos-Detections/features/val')

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64)


num_epochs = 500
checkpoint_path = '/data/npl/ICEK/License-Plate-Detection-Pipeline-with-Experiment-Tracking/Harmful-Videos-Detections/multimodal_model_less_complex_v3.pth'

train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=num_epochs, patience=4, checkpoint_path=checkpoint_path)

evaluate_model(model, val_loader, device)

