# 🎭 Real-time Facial Emotion Recognition

A professional real-time facial emotion recognition system using deep learning with CNN architecture. Detects and classifies human emotions from live webcam feeds with beautiful visualization and real-time statistics.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10+-orange.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## ✨ Features

- 🎯 **Real-time Detection**: Smooth emotion detection from webcam feed
- 📊 **Live Statistics**: Real-time FPS, session duration, emotion distribution
- 🎨 **Professional UI**: Beautiful visualization with confidence bars
- 📷 **Screenshot Capture**: Save moments with one keypress
- 🧠 **Custom CNN Model**: Train your own model with the FER2013 dataset
- 📈 **Training Analytics**: Comprehensive training visualization and metrics

## 📁 Project Structure

```
Real-time-Facial-Emotion-Recognition/
├── main.py                     # 🚀 Application entry point
├── config/
│   ├── __init__.py
│   └── settings.py             # Configuration settings
├── src/
│   ├── __init__.py
│   └── emotion_detector.py     # Main detection system
├── utils/
│   ├── __init__.py
│   ├── face_detector.py        # Face detection module
│   ├── visualizer.py           # UI visualization
│   └── data_loader.py          # Dataset utilities
├── training/
│   └── emotion_recognition_training.ipynb  # 📓 Training notebook
├── models/                     # Trained models directory
├── data/                       # Dataset directory
├── screenshots/                # Captured screenshots
├── requirements.txt
├── .gitignore
└── README.md
```

## 🎭 Supported Emotions

| Emotion | Color | Description |
|---------|-------|-------------|
| 😠 Angry | 🔴 Red | Displeasure, frustration |
| 🤢 Disgust | 🟢 Green | Strong dislike, aversion |
| 😨 Fear | 🟣 Purple | Apprehension, anxiety |
| 😊 Happy | 🟡 Yellow | Joy, satisfaction |
| 😢 Sad | 🔵 Blue | Sorrow, unhappiness |
| 😲 Surprise | 🟠 Orange | Astonishment, shock |
| 😐 Neutral | ⚪ Gray | Calm, no strong emotion |

## 🚀 Quick Start

### 1. Clone & Install

```bash
git clone https://github.com/AzizBahloul/Real-time-Facial-Emotion-Recognition.git
cd Real-time-Facial-Emotion-Recognition

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### 2. Train the Model (First Time)

Open the training notebook and run all cells:
```bash
jupyter notebook training/emotion_recognition_training.ipynb
```

The notebook will:
- Download the FER2013 dataset using `kagglehub`
- Visualize the dataset distribution
- Train a CNN model
- Save the model to `models/`

### 3. Run the Application

```bash
python main.py
```

### Command Line Options

```bash
python main.py --help

# Options:
#   --model, -m PATH    Custom model path
#   --camera, -c ID     Camera device ID (default: 0)
#   --list-cameras      List available cameras
```

## ⌨️ Controls

| Key | Action |
|-----|--------|
| `Q` | Quit application |
| `S` | Take screenshot |
| `R` | Reset statistics |

## 📊 Dataset

This project uses the **FER2013** dataset from Kaggle:
- 48x48 pixel grayscale images
- 7 emotion classes
- ~28,709 training images
- ~3,589 test images

The dataset is automatically downloaded using `kagglehub`:
```python
import kagglehub
path = kagglehub.dataset_download("msambare/fer2013")
```

## 🧠 Model Architecture

Custom CNN with 4 convolutional blocks:
- Conv2D + BatchNorm + MaxPool + Dropout (64 → 128 → 256 → 512 filters)
- Dense layers with dropout for regularization
- Softmax output for 7-class classification

## 📈 Training Features

The training notebook includes:
- 📊 Dataset visualization and class distribution
- 🔄 Data augmentation (rotation, shift, flip, zoom)
- ⚖️ Class weights for imbalanced data
- 📉 Learning rate scheduling
- 🛑 Early stopping
- 📋 Confusion matrix and classification report
- 💾 Multiple model export formats (.keras, .h5, SavedModel)

## 🛠️ Requirements

- Python 3.8+
- TensorFlow 2.10+
- OpenCV 4.5+
- NumPy, Pandas, Matplotlib, Seaborn
- kagglehub (for dataset download)
- scikit-learn (for evaluation metrics)

See `requirements.txt` for complete list.

## 📷 Screenshots

The application features:
- Real-time face detection with corner-style bounding boxes
- Emotion label with confidence percentage
- Live confidence bar chart for all emotions
- Statistics panel with FPS, session time, and emotion distribution

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- FER2013 Dataset: [Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)
- OpenCV for face detection
- TensorFlow/Keras for deep learningbash
