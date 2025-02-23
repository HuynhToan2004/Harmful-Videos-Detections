
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

whisper_model = whisper.load_model("base")  
tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model.eval() 

#=================================BiLSTM TextModel ==============================
class BiLSTMTextModel(nn.Module):
    def __init__(self, input_dim=768, hidden_dim=128, num_layers=1):
        """
        Args:
            input_dim  (int): Kích thước embedding đầu vào (VD PhoBERT: 768)
            hidden_dim (int): Số chiều ẩn của LSTM
            num_layers (int): Số lớp LSTM stack
        """
        super(BiLSTMTextModel, self).__init__()
        # LSTM hai chiều (bidirectional=True)
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,      # shape input: (batch_size, seq_len, input_dim)
            bidirectional=True
        )
        self.fc = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(0.5)
        self.relu = nn.ReLU()

    def forward(self, x):
    
        out, (h_n, c_n) = self.lstm(x)
        last_feat = out[:, -1, :]                  # (batch_size, hidden_dim * 2)
        x = self.relu(self.fc(last_feat))          # (batch_size, 128)
        x = self.dropout(x)
        return x




class VisualModel(nn.Module):
    def __init__(self, hidden_size=256, num_layers=1):
        super(VisualModel, self).__init__()
        self.cnn = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)  # Sử dụng weights mới
        # Loại bỏ lớp cuối cùng (fully connected layer)
        self.cnn = nn.Sequential(*list(self.cnn.children())[:-1])  # Output: 2048-dim
        self.lstm = nn.LSTM(input_size=2048, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        batch_size, num_frames, C, H, W = x.size()
        # Trích xuất đặc trưng từng khung hình
        x = x.view(batch_size * num_frames, C, H, W)
        with torch.no_grad():  # Không huấn luyện CNN
            features = self.cnn(x)  # Shape: (batch_size*num_frames, 2048, 1, 1)
            features = features.view(batch_size, num_frames, -1)  # Shape: (batch_size, num_frames, 2048)
        # Xử lý qua LSTM
        lstm_out, (hn, cn) = self.lstm(features)  # lstm_out: (batch_size, num_frames, hidden_size)
        return hn[-1]  # Shape: (batch_size, hidden_size)

class CrossAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(CrossAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value):
        # query, key, value: [sequence_length, batch_size, embed_dim]
        attn_output, attn_weights = self.multihead_attn(query, key, value)
        return attn_output

# Định nghĩa mô hình AudioModel (đã cập nhật với tham số max_length)
class AudioModel(nn.Module):
    def __init__(self, n_mfcc=40, max_length=40, hidden_size=128):
        super(AudioModel, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d((2,2))
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * (n_mfcc // 4) * (max_length // 4), hidden_size)  # Điều chỉnh dựa trên kích thước MFCC
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

class TextModel(nn.Module):
    def __init__(self, text_embedding_size):
        super(TextModel, self).__init__()
        self.fc = nn.Linear(text_embedding_size, 128)  # Chỉnh sửa kích thước nếu cần
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc(x))
        x = self.dropout(x)
        return x  # Shape: (batch_size, 128)

# Định nghĩa mô hình MultimodalModel với Attention và Text
class MultimodalModelWithAttention(nn.Module):
    def __init__(self, num_classes, visual_hidden_size=256, audio_hidden_size=128,
                 text_hidden_size=128, n_mfcc=40, max_length=40, embed_dim=256, num_heads=4):
        super(MultimodalModelWithAttention, self).__init__()
        self.visual_model = VisualModel(hidden_size=visual_hidden_size)
        self.audio_model = AudioModel(n_mfcc=n_mfcc, max_length=max_length, hidden_size=audio_hidden_size)

        # Nếu kích thước embed_dim của visual và audio khác nhau, thêm lớp projection
        if audio_hidden_size != visual_hidden_size:
            self.audio_projection = nn.Linear(audio_hidden_size, visual_hidden_size)
        else:
            self.audio_projection = None

        # Attention Layer
        self.cross_attention = CrossAttention(embed_dim=visual_hidden_size, num_heads=num_heads)

        # Text Model
        self.text_model = TextModel(text_embedding_size=768)  # Kích thước của PhoBERT [CLS] token
        
        # Bộ phân loại sau khi kết hợp attention và text
        self.classifier = nn.Sequential(
            nn.Linear(visual_hidden_size * 2 + text_hidden_size, 256),  # visual_hidden_size * 2 + text_hidden_size
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, visual, audio, text):
        visual_feat = self.visual_model(visual)  # Shape: (batch_size, visual_hidden_size)
        audio_feat = self.audio_model(audio)      # Shape: (batch_size, audio_hidden_size)

        # Reshape để phù hợp với MultiheadAttention: [sequence_length, batch_size, embed_dim]
        visual_feat = visual_feat.unsqueeze(0)    # Shape: (1, batch_size, visual_hidden_size)
        audio_feat = audio_feat.unsqueeze(0)      # Shape: (1, batch_size, audio_hidden_size)

        # Nếu cần, project audio_feat về kích thước embed_dim của visual_feat
        if self.audio_projection is not None:
            audio_feat = self.audio_projection(audio_feat)  # Shape: (1, batch_size, visual_hidden_size)

        # Thực hiện Cross-Attention: Visual as query, Audio as key and value
        attn_output = self.cross_attention(query=visual_feat, key=audio_feat, value=audio_feat)  # Shape: (1, batch_size, visual_hidden_size)

        attn_output = attn_output.squeeze(0)  # Shape: (batch_size, visual_hidden_size)
        combined_visual_audio = torch.cat((attn_output, audio_feat.squeeze(0)), dim=1)  # Shape: (batch_size, visual_hidden_size * 2)

        # Xử lý text
        text_feat = self.text_model(text)  # Shape: (batch_size, text_hidden_size)

        # Kết hợp tất cả đặc trưng
        combined = torch.cat((combined_visual_audio, text_feat), dim=1)  # Shape: (batch_size, visual_hidden_size * 2 + text_hidden_size)

        out = self.classifier(combined)
        return out
    












#--------------------------------------------------------VER2: more complex----------------------------------------
# import cv2
# import librosa
# import numpy as np
# import os
# from moviepy.editor import VideoFileClip
# import torch
# from torch.utils.data import Dataset, DataLoader
# from torchvision import transforms
# import torch.nn as nn
# from torchvision import models
# from torchvision.models import EfficientNet_B0_Weights
# import torch.optim as optim
# from tqdm import tqdm
# from transformers import AutoTokenizer, AutoModel
# import whisper
# import torchaudio

# # Khởi tạo các mô hình pre-trained
# whisper_model = whisper.load_model("base")  
# tokenizer = AutoTokenizer.from_pretrained("vinai/phobert-base")
# text_model = AutoModel.from_pretrained("vinai/phobert-base")
# text_model.eval()

# # Định nghĩa lớp Cross-Attention
# class CrossAttention(nn.Module):
#     def __init__(self, embed_dim, num_heads):
#         super(CrossAttention, self).__init__()
#         self.multihead_attn = nn.MultiheadAttention(embed_dim, num_heads)

#     def forward(self, query, key, value):
#         # query, key, value: [sequence_length, batch_size, embed_dim]
#         attn_output, attn_weights = self.multihead_attn(query, key, value)
#         return attn_output

# # Định nghĩa mô hình VisualModel với EfficientNet
# class VisualModel(nn.Module):
#     def __init__(self, hidden_size=256):
#         super(VisualModel, self).__init__()
#         # Sử dụng EfficientNet-B0 thay thế ResNet50
#         self.cnn = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
#         self.cnn.classifier = nn.Identity()  # Loại bỏ lớp classifier cuối cùng
#         self.lstm = nn.LSTM(input_size=1280, hidden_size=hidden_size, num_layers=1, batch_first=True)  # 1280 là output của EfficientNet-B0

#         # Freeze CNN layers để tiết kiệm tài nguyên
#         for param in self.cnn.parameters():
#             param.requires_grad = False

#     def forward(self, x):
#         batch_size, num_frames, C, H, W = x.size()
#         # Trích xuất đặc trưng từng khung hình
#         x = x.view(batch_size * num_frames, C, H, W)
#         with torch.no_grad():  # Không huấn luyện CNN
#             features = self.cnn(x)  # Shape: (batch_size*num_frames, 1280)
#         features = features.view(batch_size, num_frames, -1)  # Shape: (batch_size, num_frames, 1280)
#         # Xử lý qua LSTM
#         lstm_out, (hn, cn) = self.lstm(features)  # lstm_out: (batch_size, num_frames, hidden_size)
#         return hn[-1]  # Shape: (batch_size, hidden_size)

# # Định nghĩa mô hình AudioModel với PANNs
# class AudioModel(nn.Module):
#     def __init__(self, hidden_size=128):
#         super(AudioModel, self).__init__()
#         # Sử dụng Pretrained PANNs CNN14
#         self.panns = torchaudio.pipelines.WAV2VEC2_BASE  # Placeholder, replace with actual PANNs model if available
#         # Nếu không có PANNs, có thể sử dụng ResNet như ban đầu
#         self.conv1 = nn.Conv2d(1, 32, kernel_size=(3,3), stride=(1,1), padding=(1,1))
#         self.bn1 = nn.BatchNorm2d(32)
#         self.pool = nn.MaxPool2d((2,2))
#         self.conv2 = nn.Conv2d(32, 64, kernel_size=(3,3), stride=(1,1), padding=(1,1))
#         self.bn2 = nn.BatchNorm2d(64)
#         self.fc1 = nn.Linear(64 * 10 * 10, hidden_size)  # Điều chỉnh dựa trên kích thước MFCC sau pooling
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         # x shape: (batch_size, n_mfcc, max_length)
#         x = x.unsqueeze(1)  # Shape: (batch_size, 1, n_mfcc, max_length)
#         x = self.pool(self.relu(self.bn1(self.conv1(x))))  # Shape: (batch_size, 32, n_mfcc/2, max_length/2)
#         x = self.pool(self.relu(self.bn2(self.conv2(x))))  # Shape: (batch_size, 64, n_mfcc/4, max_length/4)
#         x = x.view(x.size(0), -1)  # Shape: (batch_size, 64 * (n_mfcc//4) * (max_length//4))
#         x = self.dropout(self.relu(self.fc1(x)))  # Shape: (batch_size, hidden_size)
#         return x  # Shape: (batch_size, hidden_size)

# # Định nghĩa mô hình TextModel
# class TextModel(nn.Module):
#     def __init__(self, text_embedding_size):
#         super(TextModel, self).__init__()
#         self.fc = nn.Linear(text_embedding_size, 128)  # Chỉnh sửa kích thước nếu cần
#         self.relu = nn.ReLU()
#         self.dropout = nn.Dropout(0.5)

#     def forward(self, x):
#         x = self.relu(self.fc(x))
#         x = self.dropout(x)
#         return x  # Shape: (batch_size, 128)

# # Định nghĩa mô hình MultimodalModel với Attention và Text
# class MultimodalModelWithAttention(nn.Module):
#     def __init__(self, num_classes, visual_hidden_size=256, audio_hidden_size=128,
#                  text_hidden_size=128, embed_dim=256, num_heads=4):
#         super(MultimodalModelWithAttention, self).__init__()
#         self.visual_model = VisualModel(hidden_size=visual_hidden_size)
#         self.audio_model = AudioModel(hidden_size=audio_hidden_size)

#         # Nếu kích thước embed_dim của visual và audio khác nhau, thêm lớp projection
#         if audio_hidden_size != visual_hidden_size:
#             self.audio_projection = nn.Linear(audio_hidden_size, visual_hidden_size)
#         else:
#             self.audio_projection = None

#         # Attention Layer
#         self.cross_attention = CrossAttention(embed_dim=visual_hidden_size, num_heads=num_heads)

#         # Text Model
#         self.text_model = TextModel(text_embedding_size=768)  # Kích thước của PhoBERT [CLS] token

#         # Thêm Cross-Attention giữa Text và Visual
#         self.cross_attention_tv = CrossAttention(embed_dim=visual_hidden_size, num_heads=num_heads)

#         # Bộ phân loại sau khi kết hợp attention và text
#         self.classifier = nn.Sequential(
#             nn.Linear(visual_hidden_size * 2 + text_hidden_size, 256),  # visual_hidden_size * 2 + text_hidden_size
#             nn.ReLU(),
#             nn.Dropout(0.5),
#             nn.Linear(256, num_classes)
#         )

#     def forward(self, visual, audio, text):
#         visual_feat = self.visual_model(visual)  # Shape: (batch_size, visual_hidden_size)
#         audio_feat = self.audio_model(audio)      # Shape: (batch_size, audio_hidden_size)

#         # Reshape để phù hợp với MultiheadAttention: [sequence_length, batch_size, embed_dim]
#         visual_feat = visual_feat.unsqueeze(0)    # Shape: (1, batch_size, visual_hidden_size)
#         audio_feat = audio_feat.unsqueeze(0)      # Shape: (1, batch_size, audio_hidden_size)

#         # Nếu cần, project audio_feat về kích thước embed_dim của visual_feat
#         if self.audio_projection is not None:
#             audio_feat = self.audio_projection(audio_feat)  # Shape: (1, batch_size, visual_hidden_size)

#         # Thực hiện Cross-Attention: Visual as query, Audio as key and value
#         attn_output_va = self.cross_attention(query=visual_feat, key=audio_feat, value=audio_feat)  # Shape: (1, batch_size, visual_hidden_size)
#         attn_output_va = attn_output_va.squeeze(0)  # Shape: (batch_size, visual_hidden_size)

#         # Xử lý text
#         text_feat = self.text_model(text)  # Shape: (batch_size, text_hidden_size)

#         # Reshape Text để phù hợp với MultiheadAttention
#         text_feat = text_feat.unsqueeze(0)  # Shape: (1, batch_size, text_hidden_size)

#         # Thực hiện Cross-Attention: Text as query, Visual as key and value
#         attn_output_tv = self.cross_attention_tv(query=text_feat, key=visual_feat, value=visual_feat)  # Shape: (1, batch_size, visual_hidden_size)
#         attn_output_tv = attn_output_tv.squeeze(0)  # Shape: (batch_size, visual_hidden_size)

#         # Kết hợp tất cả đặc trưng
#         combined_visual_audio = torch.cat((attn_output_va, audio_feat.squeeze(0)), dim=1)  # Shape: (batch_size, visual_hidden_size * 2)
#         combined = torch.cat((combined_visual_audio, attn_output_tv), dim=1)  # Shape: (batch_size, visual_hidden_size * 3)

#         # Phân loại
#         out = self.classifier(combined)
#         return out



#-------------------------------------------------Ver3: less complex--------------------




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
from torchvision.models import ResNet18_Weights  
import torch.optim as optim
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
import whisper

whisper_model = whisper.load_model("base")  

tokenizer = AutoTokenizer.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model = AutoModel.from_pretrained("/data/npl/ICEK/model/phobert-base")
text_model.eval() 

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
        
        # Thêm FC để nén xuống hidden_size (mặc định 128)
        self.fc = nn.Linear(512, hidden_size)

    def forward(self, x):
        """
        x.shape = (batch_size, num_frames, C=3, H, W)
        Ta vẫn giữ logic “mean pooling theo thời gian” để đơn giản.
        """
        batch_size, num_frames, C, H, W = x.size()
        # Gộp batch_size và num_frames để đưa vào ResNet
        x = x.view(batch_size * num_frames, C, H, W)

        # Không train ResNet => with torch.no_grad() cũng được
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
        # Giả sử input shape (batch_size, n_mfcc=40, max_length=40)
        # => Flatten ra 40*40 = 1600
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
        """
        visual: (batch_size, num_frames, 3, H, W)
        audio: (batch_size, n_mfcc, max_length)
        text:  (batch_size, 768) (embedding PhoBERT)
        """
        # 1) Trích xuất feature
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

        # Sau 2 lần pool => kích thước /4 theo cả 2 chiều (n_mfcc, max_length)
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
