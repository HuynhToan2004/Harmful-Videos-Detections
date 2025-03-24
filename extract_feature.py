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
from utils import extract_and_save_features
from vncorenlp import VnCoreNLP
device = 'cuda'
whisper_model = whisper.load_model("base",device=device)
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model.eval() 
text_model.to(device)

root_dir = '/data/npl/ICEK/News/DL/data/data/data_video/test_final' 
features_dir = '/data/npl/ICEK/News/DL/data_feature/test'

rdrsegmenter = VnCoreNLP(
    jar_path="VnCoreNLP-1.1.1.jar",
    annotators="wseg",
    max_heap_size="-Xmx500m"
)



def extract_and_save_features(video_path, features_dir, whisper_model, tokenizer, text_model, n_mfcc=40, max_length=40, num_frames=30, resize=(224, 224)):
    """
    Trích xuất và lưu trữ các đặc trưng visual, audio và text từ một video.

    Args:
        video_path (str): Đường dẫn đến video gốc.
        features_dir (str): Thư mục để lưu trữ các đặc trưng đã trích xuất.
        whisper_model: Mô hình Whisper để trích xuất transcript từ audio.
        tokenizer: Tokenizer của PhoBERT.
        text_model: Mô hình PhoBERT để trích xuất embedding văn bản.
        n_mfcc (int): Số lượng MFCC được trích xuất từ âm thanh.
        max_length (int): Độ dài tối đa của chuỗi MFCC.
        num_frames (int): Số lượng khung hình được trích xuất từ video.
        resize (tuple): Kích thước để resize các khung hình (H, W).
    """
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    os.makedirs(features_dir, exist_ok=True)

    # -----------------------------
    # Trích xuất Visual Features
    # -----------------------------
    cap = cv2.VideoCapture(video_path)
    frames = []
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total_frames // num_frames, 1)

    for i in range(0, total_frames, interval):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if ret:
            frame = cv2.resize(frame, resize)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        if len(frames) == num_frames:
            break
    cap.release()

    # Nếu không đủ số khung hình, thêm các khung hình đen
    while len(frames) < num_frames:
        frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))

    frames = np.array(frames).transpose(0, 3, 1, 2) / 255.0  # Normalize
    frames_tensor = torch.tensor(frames, dtype=torch.float)
    torch.save(frames_tensor, os.path.join(features_dir, f"{video_id}_visual.pt"))

    # -----------------------------
    # Trích xuất Audio Features
    # -----------------------------
    try:
        video = VideoFileClip(video_path)
        audio_path = os.path.join(features_dir, f"{video_id}_audio.wav")
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(audio_path, logger=None)
            y, sr = librosa.load(audio_path, sr=None)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
        
            if mfcc.shape[1] < max_length:
                pad_width = max_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_length]
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float)
            torch.save(mfcc_tensor, os.path.join(features_dir, f"{video_id}_audio.pt"))
            os.remove(audio_path)  
        else:
            # Không có âm thanh
            mfcc_tensor = torch.zeros(n_mfcc, max_length)
            torch.save(mfcc_tensor, os.path.join(features_dir, f"{video_id}_audio.pt"))
    except Exception as e:
        print(f"Error processing audio for {video_id}: {e}")
        mfcc_tensor = torch.zeros(n_mfcc, max_length)
        torch.save(mfcc_tensor, os.path.join(features_dir, f"{video_id}_audio.pt"))

    # -----------------------------
    # Trích xuất Text Features
    # -----------------------------
    try:
        video = VideoFileClip(video_path)
        audio_path = os.path.join(features_dir, f"{video_id}_audio.wav")
        audio = video.audio
        if audio is not None:
            audio.write_audiofile(audio_path, logger=None)
            result = whisper_model.transcribe(audio_path, language='vi')
            text = result['text']

            if text.strip() != "":
                segmented_sentences = rdrsegmenter.tokenize(text)
                segmented_text = " ".join(["_".join(sent) for sent in segmented_sentences])
                inputs = tokenizer(
                    segmented_text, 
                    return_tensors="pt", 
                    padding='max_length', 
                    truncation=True, 
                    max_length=128
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}

                with torch.no_grad():
                    text_output = text_model(**inputs)
                    #(shape: [batch_size=1, hidden_size=768])
                    text_embedding = text_output.last_hidden_state[:, 0, :]
            else:
              
                text_embedding = torch.zeros(1, 768).to(device)

        
            text_embedding_cpu = text_embedding.squeeze(0).cpu()
            torch.save(text_embedding_cpu, os.path.join(features_dir, f"{video_id}_text.pt"))
            os.remove(audio_path)
        else:
            text_embedding = torch.zeros(768)
            torch.save(text_embedding, os.path.join(features_dir, f"{video_id}_text.pt"))
    except Exception as e:
        print(f"Error processing text for {video_id}: {e}")
        text_embedding = torch.zeros(768)
        torch.save(text_embedding, os.path.join(features_dir, f"{video_id}_text.pt"))

os.makedirs(features_dir, exist_ok=True)

for cls in sorted(os.listdir(root_dir)):
    cls_dir = os.path.join(root_dir, cls)
    if not os.path.isdir(cls_dir):
        continue
    for file in tqdm(os.listdir(cls_dir), desc=f"Processing class {cls}"):
        if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.mov'):
            video_path = os.path.join(cls_dir, file)
            extract_and_save_features(
                video_path=video_path,
                features_dir=features_dir,
                whisper_model=whisper_model,
                tokenizer=tokenizer,
                text_model=text_model,
                n_mfcc=40,
                max_length=40,
                num_frames=30,
                resize=(224, 224)
            )
