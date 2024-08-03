
# Real-time Facial Emotion Recognition

This project implements a real-time facial emotion recognition system using Python, OpenCV, and Keras. It detects faces through a webcam feed, predicts the emotions associated with each detected face, and overlays the predicted emotion labels onto the video stream in real-time.

## Features

- **Real-time Facial Emotion Detection**: Detects and identifies emotions from live webcam feed.
- **Emotion Prediction**: Utilizes a pre-trained Keras model for predicting emotions.
- **Emotion Overlay**: Displays predicted emotion labels on the video stream.
- **Comprehensive Emotion Categories**: Recognizes seven different emotions.

## Emotion Categories

The system can detect the following emotions:
- **Angry**: Indicates displeasure or frustration.
- **Disgust**: Shows strong dislike or aversion.
- **Fear**: Represents apprehension or anxiety.
- **Happy**: Reflects joy or satisfaction.
- **Sad**: Demonstrates sorrow or unhappiness.
- **Surprise**: Shows astonishment or amazement.
- **Neutral**: Indicates a lack of strong emotional expression.

## Installation

### 1. Clone the Repository

Clone the repository using Git:
```sh
git clone https://github.com/AzizBahloul/Real-time-Facial-Emotion-Recognition.git
cd Real-time-Facial-Emotion-Recognition
```

### 2. Install Dependencies

Install the required Python libraries from `requirements.txt`:
```sh
pip install -r requirements.txt
```

Alternatively, you can manually install the dependencies using:
```sh
pip install opencv-python-headless numpy tensorflow matplotlib
```

### 3. Ensure Webcam Connectivity

Ensure your webcam is connected to your computer.

## Usage

To run the facial emotion recognition system, execute the following command:
```sh
python run_me.py
```

The system will start capturing video from the webcam and display emotion labels in real-time.

## Pre-trained Model

The pre-trained model for facial emotion recognition is included in the repository as `my_model.h5`. This model has been trained on a dataset containing facial images labeled with seven different emotions.

## Dependencies

- **OpenCV**: `opencv-python-headless` – For image and video processing.
- **NumPy**: `numpy` – For numerical operations.
- **TensorFlow**: `tensorflow` – For running the Keras model.
- **Matplotlib**: `matplotlib` – For any plotting requirements (optional).

## Contributing

Contributions are welcome. Please follow standard open-source guidelines for contributing. 

## License

This project is licensed under the [MIT License](LICENSE). See the LICENSE file for details.
