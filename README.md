# 🗣️ SAMVAAD — Real-Time Sign Language Recognition with Instant Speech

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-orange?logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-007ACC?logo=opencv&logoColor=white)
![Mediapipe](https://img.shields.io/badge/Mediapipe-Hand_Tracking-FF6F61?logo=mediapipe&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white)

---

## ✨ Project Overview
**SAMVAAD** is a real-time system that converts **hand gestures (sign language)** into **text and speech** instantly. It provides:

- ✅ Real-time hand gesture detection
- ✅ Persistent overlay text on video feed
- ✅ Instant speech output (server-side and browser-side)
- ✅ Data collection, model training, live recognition, and model deletion modes
- ✅ Clear button to reset overlay text
- ✅ Auto-speak mode: per letter or full string

This project is ideal for **bridging communication gaps** for sign language users.  

---

## 🛠️ Features

- **🎥 Real-time hand gesture detection** using Mediapipe Hands  
- **📄 Persistent overlay text** for recognized letters  
- **🔊 Immediate speech output**:
  - **Server-side**: pyttsx3  
  - **Browser-side**: gTTS  
- **📝 Data collection mode**: Capture letter samples  
- **📊 Train KNN model**: Customize `k` neighbors  
- **🖥️ Live recognition**: Detect letters and speak automatically  
- **🗑️ Delete model**: Reset training data  
- **🧹 Clear overlay text**: Reset persistent text display  

---

## ⚡ Tech Stack

| Component | Icon | Description |
|-----------|------|-------------|
| Python 🐍 | ![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python&logoColor=white) | Core programming language |
| Streamlit 🌐 | ![Streamlit](https://img.shields.io/badge/Streamlit-Interactive-orange?logo=streamlit&logoColor=white) | Web UI |
| OpenCV 📷 | ![OpenCV](https://img.shields.io/badge/OpenCV-Computer_Vision-007ACC?logo=opencv&logoColor=white) | Video processing & drawing |
| Mediapipe 🤚 | ![Mediapipe](https://img.shields.io/badge/Mediapipe-Hand_Tracking-FF6F61?logo=mediapipe&logoColor=white) | Hand landmark detection |
| NumPy 🔢 | ![NumPy](https://img.shields.io/badge/NumPy-Numerical-yellow?logo=numpy&logoColor=white) | Feature extraction & arrays |
| scikit-learn 📈 | ![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikit-learn&logoColor=white) | KNN classifier |
| pyttsx3 🗣️ | ![pyttsx3](https://img.shields.io/badge/pyttsx3-TTS-00C853) | Server-side text-to-speech |
| gTTS 🌐 | ![gTTS](https://img.shields.io/badge/gTTS-TTS-red) | Browser-side text-to-speech |

---

## 🚀 Installation

1. Clone the repository:
```bash
git clone <repo_url>
cd SAMVAAD
