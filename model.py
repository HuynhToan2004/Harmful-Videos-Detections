import cv2
import librosa
import numpy as np
import os
from moviepy.editor import VideoFileClip
import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights  



class VisualModelLessComplex(nn.Module):
    def __init__(self, hidden_size=256):
        super(VisualModelLessComplex, self).__init__()
        self.cnn = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)  
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1]) 
        self.fc = nn.Linear(512, hidden_size)  

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)
        with torch.no_grad():  
            features = self.cnn(x)  # Shape: (batch_size*num_frames, 512, 1, 1)
            features = features.view(batch_size, num_frames, -1)  # Shape: (batch_size, num_frames, 512)
    
        features = features.mean(dim=1)  # Shape: (batch_size, 512)
        features = self.fc(features)     # Shape: (batch_size, hidden_size)
        return features  # Shape: (batch_size, hidden_size)


class AudioModelLessComplex(nn.Module):
    def __init__(self, n_mfcc=40, max_length=40, hidden_size=128):
        super(AudioModelLessComplex, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * (n_mfcc // 4) * (max_length // 4), hidden_size) 
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, n_mfcc, max_length)
        x = x.unsqueeze(1)  # Shape: (batch_size, 1, n_mfcc, max_length)
        x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Shape: (batch_size, 32, n_mfcc/2, max_length/2)
        x = self.pool(self.relu(self.bn2(self.conv2(x))))  # Shape: (batch_size, 64, n_mfcc/4, max_length/4)
        x = x.view(x.size(0), -1)  # Shape: (batch_size, 64 * n_mfcc/4 * max_length/4)
        x = self.dropout(self.relu(self.fc1(x)))  # Shape: (batch_size, hidden_size)
        return x  # Shape: (batch_size, hidden_size)


class TextModelLessComplex(nn.Module):
    def __init__(self, hidden_size=128):
        super(TextModelLessComplex, self).__init__()
        self.fc = nn.Linear(768, hidden_size)  
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        return x  # Shape: (batch_size, hidden_size)

class MultimodalModel(nn.Module):
    def __init__(self, num_classes, visual_hidden_size=256, audio_hidden_size=128, 
                 text_hidden_size=128, embed_dim=256):
        super(MultimodalModel, self).__init__()
        self.visual_model = VisualModelLessComplex(hidden_size=visual_hidden_size)
        self.audio_model = AudioModelLessComplex(hidden_size=audio_hidden_size)
        self.text_model = TextModelLessComplex(hidden_size=text_hidden_size)

        # Fusion Layer
        self.fusion = nn.Linear(visual_hidden_size + audio_hidden_size + text_hidden_size, embed_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

        self.classifier = nn.Sequential(
            nn.Linear(embed_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, visual, audio, text):
        visual_feat = self.visual_model(visual)  # Shape: (batch_size, visual_hidden_size)
        audio_feat = self.audio_model(audio)      # Shape: (batch_size, audio_hidden_size)
        text_feat = self.text_model(text)        # Shape: (batch_size, text_hidden_size)

        # Kết hợp đặc trưng từ tất cả các modal
        combined = torch.cat((visual_feat, audio_feat, text_feat), dim=1)  # Shape: (batch_size, visual_hidden_size + audio_hidden_size + text_hidden_size)
        combined = self.fusion(combined)  # Shape: (batch_size, embed_dim)
        combined = self.relu(combined)
        combined = self.dropout(combined)

        # Phân loại
        out = self.classifier(combined)  # Shape: (batch_size, num_classes)
        return out






#=======================================Tiny Model: các nhãn đều có thể nhận ra, ít nhất là có thể học dc, nhãn violent hc dc nhiều nhất ========================================
import torch
import torch.nn as nn
from torchvision import models

class TinyVisualModel(nn.Module):
    def __init__(self, hidden_size=128):
        super(TinyVisualModel, self).__init__()
        # Lấy ResNet18 pretrained
        base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        # Đóng băng toàn bộ trọng số
        for param in base_model.parameters():
            param.requires_grad = False
        
        # Bỏ lớp FC cuối cùng
        # Sau lớp cuối, ResNet18 sẽ cho vector 512-dim ở đầu ra
        self.base_model = nn.Sequential(*list(base_model.children())[:-1])
        
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        x = x.view(batch_size * num_frames, C, H, W)

        with torch.no_grad():
            features = self.base_model(x)  # (batch_size*num_frames, 512, 1, 1)
        
        # Giờ reshape về (batch_size, num_frames, 512)
        features = features.view(batch_size, num_frames, 512)
        
        # Lấy trung bình theo trục num_frames
        features = features.mean(dim=1)  # (batch_size, 512)
        
        # Qua FC giảm còn hidden_size
        features = self.fc(features)     # (batch_size, hidden_size)
        return features

class TinyAudioModel(nn.Module):
    def __init__(self, n_mfcc=40, max_length=40, out_dim=128):
        super(TinyAudioModel, self).__init__()

        self.fc = nn.Linear(n_mfcc * max_length, out_dim)

    def forward(self, x):
        # x: (batch_size, n_mfcc, max_length)
        batch_size, n_mfcc, max_len = x.size()
        x = x.view(batch_size, -1)     # (batch_size, 40*40)
        x = self.fc(x)                 # (batch_size, out_dim)
        return x

class TinyTextModel(nn.Module):
    def __init__(self, in_dim=768, out_dim=128):
        super(TinyTextModel, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        # Có thể thêm ReLU, Dropout nhẹ
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        # x: (batch_size, 768) => (batch_size, out_dim)
        x = self.fc(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class TinyMultimodalModel(nn.Module):
    def __init__(self, 
                 num_classes,
                 visual_hidden_size=128,
                 audio_hidden_size=128,
                 text_hidden_size=128,
                 fusion_dim=128):
        super(TinyMultimodalModel, self).__init__()
        # Ba module con
        self.visual_model = TinyVisualModel(hidden_size=visual_hidden_size)
        self.audio_model = TinyAudioModel(out_dim=audio_hidden_size)
        self.text_model  = TinyTextModel(in_dim=768, out_dim=text_hidden_size)

        # Layer kết hợp
        # => Tổng đầu vào = visual_hidden_size + audio_hidden_size + text_hidden_size
        self.fusion = nn.Linear(visual_hidden_size + audio_hidden_size + text_hidden_size, fusion_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
        # Classifier cuối
        self.classifier = nn.Linear(fusion_dim, num_classes)

    def forward(self, visual, audio, text):

        v_feat = self.visual_model(visual)  # (batch_size, visual_hidden_size)
        a_feat = self.audio_model(audio)    # (batch_size, audio_hidden_size)
        t_feat = self.text_model(text)      # (batch_size, text_hidden_size)

        # 2) Nối lại
        combined = torch.cat([v_feat, a_feat, t_feat], dim=1)
        combined = self.fusion(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)

        # 3) Phân loại
        out = self.classifier(combined)
        return out



#==============================================SIMPLY MODEL: chỉ học dc violent, các cái còn lại phế=========================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision.models import SqueezeNet1_0_Weights


class SimpleVideoModel(nn.Module):
    def __init__(self, hidden_size=128):
        super(SimpleVideoModel, self).__init__()
        self.backbone = models.squeezenet1_0(weights=SqueezeNet1_0_Weights.IMAGENET1K_V1)
        self.features = self.backbone.features  # output shape (batch, 512, H/?, W/?)
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        
        x = x.mean(dim=1)

        # Qua SqueezeNet
        with torch.no_grad():
            feat = self.features(x)    # (batch_size, 512, H/16, W/16) tuỳ input
        feat = self.avgpool(feat)      # (batch_size, 512, 1, 1)
        feat = feat.view(batch_size, 512)
        
        # Giảm chiều
        feat = self.fc(feat)           # (batch_size, hidden_size)
        return feat

# --------------------------------------
# 2) Audio model: 2 lớp Conv đơn giản
# --------------------------------------
class SimpleAudioModel(nn.Module):
    def __init__(self, n_mfcc=40, max_length=40, hidden_size=64):
        super(SimpleAudioModel, self).__init__()
        # Chỉ 2 lớp conv nho nhỏ
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2)

        # => out_channels=32 => Flatten => FC
        self.fc = nn.Linear(32 * (n_mfcc//4) * (max_length//4), hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, n_mfcc, max_length)
        x = x.unsqueeze(1)  # -> (batch_size, 1, n_mfcc, max_length)
        x = F.relu(self.bn1(self.conv1(x)))  # -> (batch_size, 16, n_mfcc, max_length)
        x = self.pool(x)                     # -> (batch_size, 16, n_mfcc/2, max_length/2)

        x = F.relu(self.bn2(self.conv2(x)))  # -> (batch_size, 32, n_mfcc/2, max_length/2)
        x = self.pool(x)                     # -> (batch_size, 32, n_mfcc/4, max_length/4)

        x = x.view(x.size(0), -1)            # Flatten
        x = self.dropout(F.relu(self.fc(x))) # -> (batch_size, hidden_size)
        return x

# --------------------------------------
# 3) Text model: FC 768->64
# --------------------------------------
class SimpleTextModel(nn.Module):
    def __init__(self, input_dim=768, hidden_size=64):
        super(SimpleTextModel, self).__init__()
        self.fc = nn.Linear(input_dim, hidden_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # x shape: (batch_size, 768) [CLS] token
        x = F.relu(self.fc(x))         # -> (batch_size, hidden_size)
        x = self.dropout(x)
        return x

# --------------------------------------
# 4) Multimodal Model
# --------------------------------------
class SimpleMultimodalModel(nn.Module):
    def __init__(self, num_classes=5, 
                 visual_hidden_size=128,
                 audio_hidden_size=64,
                 text_hidden_size=64):
        super(SimpleMultimodalModel, self).__init__()
        self.visual_model = SimpleVideoModel(hidden_size=visual_hidden_size)
        self.audio_model  = SimpleAudioModel(hidden_size=audio_hidden_size)
        self.text_model   = SimpleTextModel(hidden_size=text_hidden_size)

        # Fuse: concat (visual + audio + text) => linear => output
        fusion_dim = visual_hidden_size + audio_hidden_size + text_hidden_size
        self.fusion_fc = nn.Linear(fusion_dim, 128)
        self.classifier = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, visual, audio, text):
        v_feat = self.visual_model(visual)  # (batch_size, 128)
        a_feat = self.audio_model(audio)    # (batch_size, 64)
        t_feat = self.text_model(text)      # (batch_size, 64)

        combined = torch.cat([v_feat, a_feat, t_feat], dim=1)  # (batch_size, 128+64+64=256)
        fused    = self.fusion_fc(combined)                    # (batch_size, 128)
        out      = self.classifier(fused)                      # (batch_size, num_classes)
        return out
