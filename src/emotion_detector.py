"""
Real-time Facial Emotion Recognition System
Professional emotion detector with real-time statistics and visualization
"""

import cv2
import numpy as np
import time
from pathlib import Path
from datetime import datetime
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config.settings import (
    EMOTIONS, EMOTION_COLORS, MODEL_PATH, MODEL_INPUT_SIZE,
    WINDOW_WIDTH, WINDOW_HEIGHT, CONFIDENCE_THRESHOLD
)
from utils.face_detector import FaceDetector
from utils.visualizer import EmotionVisualizer


class EmotionRecognitionSystem:
    """
    Professional real-time facial emotion recognition system.
    
    Features:
    - Real-time face detection using Haar Cascade
    - Emotion classification using CNN model
    - Professional UI with statistics
    - Screenshot capability
    - Session statistics tracking
    """
    
    def __init__(self, model_path: str = None, camera_id: int = 0):
        """
        Initialize the emotion recognition system.
        
        Args:
            model_path: Path to the trained model
            camera_id: Camera device ID
        """
        self.camera_id = camera_id
        self.model_path = model_path or str(MODEL_PATH)
        self.model = None
        self.cap = None
        self.running = False
        
        # Initialize components
        self.face_detector = FaceDetector()
        self.visualizer = EmotionVisualizer(EMOTIONS, EMOTION_COLORS)
        
        # Session data
        self.screenshots_dir = Path("screenshots")
        self.screenshots_dir.mkdir(exist_ok=True)
    
    def load_model(self):
        """Load the trained emotion recognition model."""
        from tensorflow.keras.models import load_model
        
        print(f"Loading model from: {self.model_path}")
        
        # Try different model formats
        model_paths = [
            self.model_path,
            str(Path(self.model_path).with_suffix('.keras')),
            str(Path(self.model_path).with_suffix('.h5')),
            str(Path(self.model_path).parent / 'emotion_model.keras'),
            str(Path(self.model_path).parent / 'emotion_model.h5'),
            str(Path(self.model_path).parent / 'my_model.h5'),
        ]
        
        for path in model_paths:
            if Path(path).exists():
                try:
                    self.model = load_model(path)
                    print(f"✅ Model loaded successfully from: {path}")
                    return True
                except Exception as e:
                    print(f"⚠️ Failed to load from {path}: {e}")
        
        print("❌ No valid model found. Please train a model first.")
        print("   Run the training notebook: training/emotion_recognition_training.ipynb")
        return False
    
    def initialize_camera(self) -> bool:
        """Initialize video capture."""
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print("❌ Failed to open camera")
            return False
        
        # Set camera properties for better performance
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("✅ Camera initialized successfully")
        return True
    
    def process_frame(self, frame: np.ndarray) -> tuple:
        """
        Process a single frame for emotion detection.
        
        Args:
            frame: Input BGR frame
            
        Returns:
            Tuple of (processed_frame, detections)
        """
        # Detect faces
        faces, gray = self.face_detector.detect_faces(frame)
        detections = []
        
        for bbox in faces:
            x, y, w, h = bbox
            
            # Extract and preprocess face ROI
            roi = self.face_detector.extract_face_roi(gray, bbox, MODEL_INPUT_SIZE)
            processed_roi = self.face_detector.preprocess_for_model(roi)
            
            # Predict emotion
            predictions = self.model.predict(processed_roi, verbose=0)[0]
            emotion_idx = np.argmax(predictions)
            confidence = predictions[emotion_idx]
            emotion = EMOTIONS[emotion_idx]
            
            # Store detection
            detections.append({
                'bbox': bbox,
                'emotion': emotion,
                'confidence': confidence,
                'predictions': predictions
            })
            
            # Update statistics
            if confidence >= CONFIDENCE_THRESHOLD:
                self.visualizer.update_stats(emotion, confidence)
            
            # Draw face box and label
            frame = self.visualizer.draw_face_box(frame, bbox, emotion, confidence)
            
            # Draw confidence bars for first face
            if len(detections) == 1:
                frame = self.visualizer.draw_confidence_bar(
                    frame, predictions, 10, 10
                )
        
        return frame, detections
    
    def take_screenshot(self, frame: np.ndarray):
        """Save a screenshot of the current frame."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = self.screenshots_dir / f"emotion_capture_{timestamp}.jpg"
        cv2.imwrite(str(filename), frame)
        print(f"📸 Screenshot saved: {filename}")
    
    def run(self):
        """Main loop for real-time emotion recognition."""
        if not self.load_model():
            return
        
        if not self.initialize_camera():
            return
        
        self.running = True
        print("\n" + "=" * 60)
        print("🎭 Real-time Facial Emotion Recognition")
        print("=" * 60)
        print("Controls:")
        print("  Q - Quit")
        print("  S - Screenshot")
        print("  R - Reset statistics")
        print("=" * 60 + "\n")
        
        try:
            while self.running:
                ret, frame = self.cap.read()
                
                if not ret:
                    print("⚠️ Failed to read frame")
                    continue
                
                # Flip frame for mirror effect
                frame = cv2.flip(frame, 1)
                
                # Process frame
                frame, detections = self.process_frame(frame)
                
                # Calculate and display FPS
                fps = self.visualizer.calculate_fps()
                
                # Draw stats panel
                frame = self.visualizer.draw_stats_panel(frame, fps, len(detections))
                
                # Draw instructions
                frame = self.visualizer.draw_instructions(frame)
                
                # Resize for display
                display_frame = cv2.resize(frame, (WINDOW_WIDTH, WINDOW_HEIGHT))
                
                # Show frame
                cv2.imshow('Emotion Recognition', display_frame)
                
                # Handle keyboard input
                key = cv2.waitKey(1) & 0xFF
                
                if key == ord('q') or key == ord('Q'):
                    print("\n👋 Quitting...")
                    self.running = False
                
                elif key == ord('s') or key == ord('S'):
                    self.take_screenshot(frame)
                
                elif key == ord('r') or key == ord('R'):
                    self.visualizer.reset_session_stats()
                    print("🔄 Statistics reset")
        
        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        
        finally:
            self.cleanup()
    
    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()
        
        # Print session summary
        print("\n" + "=" * 60)
        print("📊 Session Summary")
        print("=" * 60)
        print(f"Total detections: {self.visualizer.total_detections}")
        if self.visualizer.total_detections > 0:
            print("\nEmotion distribution:")
            for emotion, count in self.visualizer.session_emotions.items():
                pct = count / self.visualizer.total_detections * 100
                bar = "█" * int(pct / 5)
                print(f"  {emotion:10s}: {bar} {pct:.1f}%")
        print("=" * 60)


def main():
    """Entry point for the emotion recognition system."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Real-time Facial Emotion Recognition")
    parser.add_argument('--model', type=str, default=None,
                        help='Path to the trained model')
    parser.add_argument('--camera', type=int, default=0,
                        help='Camera device ID (default: 0)')
    
    args = parser.parse_args()
    
    # Create and run the system
    system = EmotionRecognitionSystem(
        model_path=args.model,
        camera_id=args.camera
    )
    system.run()


if __name__ == "__main__":
    main()
