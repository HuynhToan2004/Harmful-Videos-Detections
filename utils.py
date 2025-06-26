import cv2
import librosa
import numpy as np
import os
os.environ["IMAGEIO_FFMPEG_EXE"] = "/Harmful-Videos-Detections/ffmpeg/bin/ffmpeg"
from moviepy.editor import VideoFileClip
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from transformers import AutoTokenizer, AutoModel
import whisper
import tempfile

whisper_model = whisper.load_model("large")  
tokenizer = AutoTokenizer.from_pretrained("/assets/phobert-base")
text_model = AutoModel.from_pretrained("/assets/phobert-base")
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



class PreExtractedFeatureDataset(Dataset):
    def __init__(self, feature_dir):
        self.feature_paths = []
        self.labels = []
        self.class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(feature_dir)))}
        
        for cls in sorted(os.listdir(feature_dir)):
            cls_path = os.path.join(feature_dir, cls)
            for file in os.listdir(cls_path):
                if file.endswith('.pt'):
                    self.feature_paths.append(os.path.join(cls_path, file))
                    self.labels.append(self.class_to_idx[cls])

    def __len__(self):
        return len(self.feature_paths)

    def __getitem__(self, idx):
        feature = torch.load(self.feature_paths[idx])
        frames = feature['frames']
        mfcc = feature['mfcc']
        text_embedding = feature['text_embedding']
        label = self.labels[idx]
        return frames, mfcc, text_embedding, label



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

