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
import tempfile

whisper_model = whisper.load_model("base")  
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model.eval() 


def extract_frames(video_path, num_frames=30, resize=(224, 224)):
    """
    Trích xuất các khung hình từ video.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return np.zeros((num_frames, resize[0], resize[1], 3))

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

    while len(frames) < num_frames:
        frames.append(np.zeros((resize[0], resize[1], 3), dtype=np.uint8))

    return np.array(frames)



def extract_audio_mfcc_and_text(video_path, n_mfcc=40, max_length=40):
    """
    Trích xuất MFCC và văn bản từ âm thanh của video bằng MoviePy và Whisper.
    Nếu không thể trích xuất âm thanh hoặc văn bản, trả về MFCC zero và text rỗng.
    """
    # Giá trị mặc định
    mfcc = np.zeros((n_mfcc, max_length))
    text = ""

    # Kiểm tra file video có tồn tại
    if not os.path.isfile(video_path):
        print(f"Video file {video_path} does not exist.")
        return mfcc, text

    # Tạo file âm thanh tạm (dùng NamedTemporaryFile)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmpfile:
        audio_path = tmpfile.name

    try:
       
        video = VideoFileClip(video_path)
        audio = video.audio
        if audio is None:
            raise ValueError(f"No audio track found in {video_path}")

        
        audio.write_audiofile(audio_path, logger=None)

        if not os.path.isfile(audio_path):
            raise FileNotFoundError(f"Failed to create audio file for {video_path}")

        y, sr = librosa.load(audio_path, sr=None)
        mfcc_raw = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)

        if mfcc_raw.shape[1] < max_length:
            pad_width = max_length - mfcc_raw.shape[1]
            mfcc_raw = np.pad(mfcc_raw, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfcc_raw = mfcc_raw[:, :max_length]
        mfcc = mfcc_raw

        result = whisper_model.transcribe(audio_path, language='vi')  
        text = result['text']

    except Exception as e:
        print(f"Error processing {video_path}: {e}")

    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)

    return mfcc, text


class ToTensorNormalize:
    def __init__(self, mean, std):
        self.normalize = transforms.Normalize(mean=mean, std=std)

    def __call__(self, frames):
        # frames: tensor of shape (num_frames, C, H, W)
        frames = self.normalize(frames)
        return frames

class VideoDataset(Dataset):
    def __init__(self, root_dir, num_frames=30, n_mfcc=40, max_length=40, transform=None, tokenizer=None, text_model=None, max_text_length=128):
        self.root_dir = root_dir
        self.num_frames = num_frames
        self.n_mfcc = n_mfcc
        self.max_length = max_length
        self.transform = transform
        self.tokenizer = tokenizer
        self.text_model = text_model
        self.max_text_length = max_text_length
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.video_paths = []
        self.labels = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.endswith('.mp4') or file.endswith('.avi') or file.endswith('.mov'):
                    self.video_paths.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]

        # Trích xuất khung hình
        frames = extract_frames(video_path, num_frames=self.num_frames)
        if self.transform:
            
            frames = np.transpose(frames, (0, 3, 1, 2))  # (num_frames, H, W, C) -> (num_frames, C, H, W)
            frames = torch.tensor(frames, dtype=torch.float) / 255.0  # Scale to [0,1]
            frames = self.transform(frames)  # Apply Normalize
        else:
           
            frames = np.transpose(frames, (0, 3, 1, 2))  # (num_frames, H, W, C) -> (num_frames, C, H, W)
            frames = torch.tensor(frames, dtype=torch.float) / 255.0  # Scale to [0,1]

        # Trích xuất MFCC và Text
        mfcc, text = extract_audio_mfcc_and_text(video_path, n_mfcc=self.n_mfcc, max_length=self.max_length)
        mfcc = torch.tensor(mfcc, dtype=torch.float)

        # Xử lý Text
        if text_model is not None and tokenizer is not None and text != "":
            inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=self.max_text_length)
            with torch.no_grad():
                text_features = text_model(**inputs)
                # Lấy lớp [CLS] token
                text_embedding = text_features.last_hidden_state[:, 0, :]  # Shape: (1, hidden_size)
                text_embedding = text_embedding.squeeze(0)  # Shape: (hidden_size,)
        else:
            
            if text_model is not None:
                text_embedding = torch.zeros(text_model.config.hidden_size)
            else:
                text_embedding = torch.zeros(768)  # Kích thước mặc định của PhoBERT [CLS] token

        return frames, mfcc, text_embedding, label

transform = ToTensorNormalize(mean=[0.485, 0.456, 0.406], 
                             std=[0.229, 0.224, 0.225])






class NormalizeVideoFrames:
    """
    Áp dụng Normalize(mean, std) cho tensor video (num_frames, C, H, W).
    """
    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean).view(-1,1,1)  # shape (C,1,1)
        self.std  = torch.as_tensor(std).view(-1,1,1)

    def __call__(self, video_tensor):

        for i in range(video_tensor.size(0)):
            video_tensor[i] = (video_tensor[i] - self.mean) / self.std
        return video_tensor


normalize_transform = NormalizeVideoFrames(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)


class VideoDatasetAvailabelFeature(Dataset):
    def __init__(self, root_dir, features_dir, transform=normalize_transform):
        """
        root_dir: chứa video gốc (chỉ để lấy danh sách + label).
        features_dir: thư mục chứa file .pt (xxx_visual.pt, xxx_audio.pt, xxx_text.pt).
        transform: gọi để thực hiện Normalize/augment trên visual_feat (nếu cần).
        """
        self.root_dir = root_dir
        self.features_dir = features_dir
        self.transform = transform
        
        self.classes = sorted(os.listdir(root_dir))
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.classes)}
        self.video_paths = []
        self.labels = []

        for cls in self.classes:
            cls_dir = os.path.join(root_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for file in os.listdir(cls_dir):
                if file.endswith(('.mp4', '.avi', '.mov')):
                    self.video_paths.append(os.path.join(cls_dir, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.video_paths)

    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        video_id = os.path.splitext(os.path.basename(video_path))[0]

        # Load visual
        visual_pt = os.path.join(self.features_dir, f"{video_id}_visual.pt")
        visual_feat = torch.load(visual_pt)  
    
        if self.transform:
            visual_feat = self.transform(visual_feat)

        # Load audio (MFCC)
        audio_pt = os.path.join(self.features_dir, f"{video_id}_audio.pt")
        audio_feat = torch.load(audio_pt)
        

        
        text_pt = os.path.join(self.features_dir, f"{video_id}_text.pt")
        text_feat = torch.load(text_pt)

        return visual_feat, audio_feat, text_feat, label

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

    frames = np.array(frames).transpose(0, 3, 1, 2) / 255.0  
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
            # Đảm bảo rằng MFCC có đúng độ dài
            if mfcc.shape[1] < max_length:
                pad_width = max_length - mfcc.shape[1]
                mfcc = np.pad(mfcc, pad_width=((0, 0), (0, pad_width)), mode='constant')
            else:
                mfcc = mfcc[:, :max_length]
            mfcc_tensor = torch.tensor(mfcc, dtype=torch.float)
            torch.save(mfcc_tensor, os.path.join(features_dir, f"{video_id}_audio.pt"))
            os.remove(audio_path)  # Xóa file WAV sau khi trích xuất MFCC
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
                inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=128)
                with torch.no_grad():
                    text_embedding = text_model(**inputs).last_hidden_state[:, 0, :]  # [CLS] token
            else:
                text_embedding = torch.zeros(768)
            torch.save(text_embedding.squeeze(0), os.path.join(features_dir, f"{video_id}_text.pt"))
            os.remove(audio_path) 
        else:
            text_embedding = torch.zeros(768)
            torch.save(text_embedding, os.path.join(features_dir, f"{video_id}_text.pt"))
    except Exception as e:
        print(f"Error processing text for {video_id}: {e}")
        text_embedding = torch.zeros(768)
        torch.save(text_embedding, os.path.join(features_dir, f"{video_id}_text.pt"))