import cv2
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Harmful-Videos-Detections/ffmpeg/bin/ffmpeg"
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
from sklearn.metrics import classification_report

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device: ",device)
whisper_model = whisper.load_model("large",device=device)  
tokenizer = AutoTokenizer.from_pretrained("/assets/phobert-base")
text_model = AutoModel.from_pretrained("/assets/phobert-base")
text_model.eval() 
text_model.to(device)

print("Bản ít phức tạp nhưng đạt hiệu suất cao nhất: ")

num_classes = len(os.listdir('/Harmful-Videos-Detections/features/train')) 

model = MultimodalModel(
    num_classes=num_classes, 
    visual_hidden_size=256, 
    audio_hidden_size=128, 
    text_hidden_size=128, 
    embed_dim=256
)

model = model.to(device)
checkpoint_path = '/Harmful-Videos-Detections/multimodal_model_less_complex_v3.pth'
checkpoint = torch.load(checkpoint_path, map_location=device)
model.load_state_dict(checkpoint)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=1e-4)

transform = ToTensorNormalize(mean=[0.485, 0.456, 0.406],  
                             std=[0.229, 0.224, 0.225])


dataset = PreExtractedFeatureDataset('/Harmful-Videos-Detections/features/test')
test = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)  

accuracy, predicted, labels = evaluate_model(model, test, device)

print("Classification Report của test_final:")
print(classification_report(
    labels,
    predicted,
    target_names=["horrible", "normal", "offensive", "pornographic",'aupertitious' ,"violent"]
))