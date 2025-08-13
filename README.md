# Real-time Facial Emotion Recognition

A real-time facial emotion recognition system that detects and classifies human emotions from live webcam feeds using computer vision and deep learning.

![Python](https://img.shields.io/badge/Python-3.7+-blue.svg)
![OpenCV](https://img.shields.io/badge/OpenCV-4.0+-green.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0+-orange.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Overview

This project implements a real-time facial emotion recognition system using Python, OpenCV, and Keras. The system captures video from your webcam, detects faces in real-time, and predicts emotions using a pre-trained deep learning model. Emotion labels are overlayed directly onto the video stream for immediate feedback.

## Features

✨ **Real-time Processing**: Detects and classifies emotions from live webcam feed
🎯 **High Accuracy**: Uses a pre-trained Keras model for reliable emotion prediction
📹 **Live Overlay**: Displays predicted emotions directly on the video stream
🎭 **7 Emotion Categories**: Recognizes a comprehensive range of human emotions
⚡ **Optimized Performance**: Efficient processing for smooth real-time operation

## Supported Emotions

The system recognizes seven distinct emotional states:

| Emotion | Description |
|---------|-------------|
| 😠 **Angry** | Displeasure, frustration, or irritation |
| 🤢 **Disgust** | Strong dislike, revulsion, or aversion |
| 😨 **Fear** | Apprehension, anxiety, or worry |
| 😊 **Happy** | Joy, satisfaction, or contentment |
| 😢 **Sad** | Sorrow, unhappiness, or melancholy |
| 😲 **Surprise** | Astonishment, amazement, or shock |
| 😐 **Neutral** | Lack of strong emotional expression |

## Requirements

### System Requirements
- Python 3.7 or higher
- Webcam (built-in or external USB camera)
- Windows, macOS, or Linux operating system

### Hardware Recommendations
- **CPU**: Intel i5 or equivalent (for real-time processing)
- **RAM**: 4GB minimum, 8GB recommended
- **Storage**: 500MB free space

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AzizBahloul/Real-time-Facial-Emotion-Recognition.git
cd Real-time-Facial-Emotion-Recognition
```

### 2. Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv emotion_recognition_env

# Activate virtual environment
# On Windows:
emotion_recognition_env\Scripts\activate
# On macOS/Linux:
source emotion_recognition_env/bin/activate
```

### 3. Install Dependencies
```bash
# Install from requirements file
pip install -r requirements.txt

# Or install manually
pip install opencv-python-headless numpy tensorflow matplotlib
```

### 4. Verify Installation
```bash
python -c "import cv2, numpy, tensorflow; print('All dependencies installed successfully!')"
```

## Usage

### Quick Start
```bash
python run_me.py
```

### Controls
- **ESC** or **Q**: Quit the application
- **SPACE**: Pause/Resume video feed (if implemented)
- **S**: Save current frame with predictions (if implemented)

### Expected Output
- A window will open displaying your webcam feed
- Detected faces will be highlighted with bounding boxes
- Emotion predictions will be displayed above each detected face
- Confidence scores may be shown alongside emotion labels

## Project Structure

```
Real-time-Facial-Emotion-Recognition/
│
├── run_me.py              # Main execution script
├── my_model.h5            # Pre-trained emotion recognition model
├── requirements.txt       # Python dependencies
├── README.md             # Project documentation
├── LICENSE               # MIT License file
│
├── data/                 # Dataset files (if included)
├── models/               # Model training scripts (if included)
└── utils/                # Utility functions (if included)
```

## Model Information

### Pre-trained Model Details
- **File**: `my_model.h5`
- **Architecture**: Convolutional Neural Network (CNN)
- **Training Dataset**: FER-2013 or similar emotion dataset
- **Input Size**: 48x48 grayscale images
- **Output**: 7 emotion classes with confidence scores

### Model Performance
The model has been trained to achieve optimal accuracy across all emotion categories. Performance may vary based on lighting conditions, face angle, and image quality.

## Troubleshooting

### Common Issues

**1. Camera Access Error**
```
Error: Could not access camera
```
- Ensure your webcam is connected and not being used by another application
- Try changing the camera index in the code (0, 1, 2, etc.)

**2. Model Loading Error**
```
Error: Could not load model file
```
- Verify that `my_model.h5` exists in the project directory
- Ensure TensorFlow is properly installed

**3. Poor Detection Performance**
- Ensure adequate lighting conditions
- Position your face clearly in front of the camera
- Avoid extreme angles or occlusions

**4. Slow Performance**
- Close other resource-intensive applications
- Consider reducing video resolution in the code
- Ensure your system meets the minimum requirements

### Getting Help
If you encounter issues:
1. Check the [Issues](https://github.com/AzizBahloul/Real-time-Facial-Emotion-Recognition/issues) page
2. Create a new issue with detailed error messages
3. Include your system specifications and Python version

## Contributing

Contributions are welcome! Here's how you can help:

### Types of Contributions
- 🐛 **Bug Reports**: Report issues or unexpected behavior
- 💡 **Feature Requests**: Suggest new features or improvements
- 📝 **Documentation**: Improve README, comments, or documentation
- 🔧 **Code Contributions**: Submit pull requests with improvements

### Development Setup
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes and test thoroughly
4. Commit your changes (`git commit -m 'Add amazing feature'`)
5. Push to the branch (`git push origin feature/amazing-feature`)
6. Open a Pull Request

### Code Style
- Follow PEP 8 guidelines
- Add comments for complex functions
- Include docstrings for all functions and classes
- Test your changes before submitting

## Technical Details

### Dependencies Overview
| Package | Version | Purpose |
|---------|---------|---------|
| OpenCV | 4.0+ | Computer vision and image processing |
| NumPy | 1.19+ | Numerical operations and array handling |
| TensorFlow | 2.0+ | Deep learning model execution |
| Matplotlib | 3.0+ | Plotting and visualization (optional) |

### Performance Optimization
- Uses OpenCV's optimized face detection algorithms
- Implements efficient preprocessing for real-time performance
- Utilizes TensorFlow's optimized inference engine

## Future Enhancements

- [ ] Support for multiple face detection simultaneously
- [ ] Emotion intensity scoring (0-100%)
- [ ] Historical emotion tracking and analysis
- [ ] Support for video file input (not just webcam)
- [ ] Mobile app integration
- [ ] Custom model training interface
- [ ] Real-time emotion statistics and reporting

## Changelog

### Version 1.0.0
- Initial release with basic emotion recognition
- Support for 7 emotion categories
- Real-time webcam processing
- Pre-trained model integration

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

### MIT License Summary
- ✅ Commercial use allowed
- ✅ Modification allowed
- ✅ Distribution allowed
- ✅ Private use allowed
- ❌ No liability or warranty

## Acknowledgments

- Thanks to the creators of the FER-2013 dataset
- OpenCV community for excellent computer vision tools
- TensorFlow team for the deep learning framework
- All contributors who help improve this project

 
---

⭐ **If you found this project helpful, please consider giving it a star!** ⭐
