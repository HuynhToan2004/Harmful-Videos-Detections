
---

# **Vietnamese Harmful Video Detection using Deep Learning**
🚀 **Multimodal Model for Detecting Vietnamese Harmful Videos**

## 📌 **Project Description**
This project leverages deep learning to detect and classify harmful video content using the **HarmfulVideosVN2023** dataset. The model is trained to categorize videos into five classes: **pornographic, offensive, violent, horrible**, and **normal**.

The model achieved an **accuracy of 83.12%** on the test set.

---

## 🗂 **Dataset**
The **HarmfulVideosVN2023** dataset consists of various video samples containing both harmful and normal content. The dataset is classified into the following categories:

| **Label**        | **Description** |
|-----------------|----------------------------------|
| **pornographic** | Pornographic content |
| **normal**       | Safe and normal content |
| **offensive**    | Offensive and hurtful content |
| **horrible**     | Horror and disturbing content |
| **violent**      | Violent and aggressive content |

---

## 📊 **Model Performance**
The model is evaluated using **Precision, Recall, and F1-score** on the test set:

| **Class**       | **Precision** | **Recall** | **F1-score** | **Support** |
|---------------|-------------|------------|-------------|------------|
| **pornographic** | 0.65 | 0.68 | 0.67 | 22 |
| **normal** | 0.79 | 0.87 | 0.83 | 31 |
| **offensive** | 0.83 | 0.97 | 0.90 | 31 |
| **horrible** | 1.00 | 0.81 | 0.90 | 32 |
| **violent** | 0.86 | 0.79 | 0.82 | 38 |

### 🎯 **Overall Performance**
- **Accuracy:** **83.12%**
- **Macro F1-score:** **82%**

📌 *The model performs well for "offensive" and "horrible" categories but requires improvement for the "pornographic" class.*

---

## ⚙️ **Installation & Usage**
### 1️⃣ **System Requirements**
- Python 3.8+
- NVIDIA GPU (recommended)

### 2️⃣ **Install Dependencies**
Run the following command to install required libraries:
```sh
pip install -r requirements.txt
```

### 3️⃣ **Train the Model**
```sh
python best_model.py --dataset HarmfulVideosVN2023
```

---


## 📌 **Technologies Used**
- **Deep Learning Frameworks:** PyTorch, Transformers  
- **Pre-trained Models:**  
  - **Whisper** (for speech-to-text transcription)  
  - **PhoBERT** (for Vietnamese text processing)  
  - **ResNet50** (for video frame feature extraction)  
- **Computer Vision:** OpenCV, MoviePy  
- **Audio Processing:** Librosa  
- **Training Optimization:** Adam Optimizer, PyTorch DataLoader  
- **Evaluation Metrics:** Precision, Recall, F1-score  

---

## 🚀 **Future Improvements**
- 🛠 **Improve model accuracy** through hyperparameter tuning.

---

## 🤝 **Contributions**
We welcome contributions! To contribute:
1. **Fork this repository**
2. **Create a new branch:** `git checkout -b feature-new`
3. **Commit your changes:** `git commit -m "Add new feature"`
4. **Push to GitHub:** `git push origin feature-new`
5. **Create a Pull Request**

---
