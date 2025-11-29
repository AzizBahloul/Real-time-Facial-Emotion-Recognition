#!/usr/bin/env python3
"""
🎭 Real-time Facial Emotion Recognition System
==============================================

A professional facial emotion recognition application using deep learning.

Features:
- Real-time emotion detection from webcam
- 7 emotion classes: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
- Professional UI with confidence visualization
- Real-time statistics tracking
- Screenshot capability

Usage:
    python main.py [--model PATH] [--camera ID]
    
Controls:
    Q - Quit application
    S - Take screenshot
    R - Reset statistics

Author: Facial Emotion Recognition Project
"""

import sys
import os
from pathlib import Path

# Ensure the project root is in the path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def check_dependencies():
    """Check if all required dependencies are installed."""
    required = {
        'cv2': 'opencv-python',
        'numpy': 'numpy',
        'tensorflow': 'tensorflow',
    }
    
    missing = []
    for module, package in required.items():
        try:
            __import__(module)
        except ImportError:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\n💡 Install with: pip install -r requirements.txt")
        return False
    
    return True


def print_banner():
    """Print application banner."""
    banner = """
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║   🎭 Real-time Facial Emotion Recognition System 🎭       ║
    ║                                                           ║
    ║   Detecting: Angry | Disgust | Fear | Happy               ║
    ║              Sad | Surprise | Neutral                     ║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """
    print(banner)


def main():
    """Main entry point for the application."""
    print_banner()
    
    # Check dependencies
    if not check_dependencies():
        sys.exit(1)
    
    # Import and run the emotion recognition system
    try:
        from src.emotion_detector import EmotionRecognitionSystem
        from config.settings import MODEL_PATH, MODELS_DIR, create_directories
        
        # Ensure directories exist
        create_directories()
        
        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(
            description="Real-time Facial Emotion Recognition System",
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog="""
Examples:
  python main.py                    # Run with default settings
  python main.py --camera 1         # Use camera device 1
  python main.py --model path/to/model.keras  # Use custom model
            """
        )
        parser.add_argument(
            '--model', '-m',
            type=str,
            default=None,
            help='Path to trained model (default: models/emotion_model.keras)'
        )
        parser.add_argument(
            '--camera', '-c',
            type=int,
            default=0,
            help='Camera device ID (default: 0)'
        )
        parser.add_argument(
            '--list-cameras',
            action='store_true',
            help='List available cameras and exit'
        )
        
        args = parser.parse_args()
        
        # List cameras if requested
        if args.list_cameras:
            import cv2
            print("\n📷 Checking available cameras...")
            available = []
            for i in range(10):
                cap = cv2.VideoCapture(i)
                if cap.isOpened():
                    available.append(i)
                    cap.release()
            
            if available:
                print(f"   Available camera IDs: {available}")
            else:
                print("   No cameras found!")
            return
        
        # Determine model path
        model_path = args.model
        if model_path is None:
            # Check for existing models
            possible_paths = [
                MODELS_DIR / 'emotion_model.keras',
                MODELS_DIR / 'emotion_model.h5',
                MODELS_DIR / 'emotion_model_best.keras',
                PROJECT_ROOT / 'my_model.h5',
            ]
            
            for path in possible_paths:
                if path.exists():
                    model_path = str(path)
                    break
            
            if model_path is None:
                print("⚠️  No trained model found!")
                print("\n📝 To train a model:")
                print("   1. Open training/emotion_recognition_training.ipynb")
                print("   2. Run all cells to train the model")
                print("   3. The model will be saved to models/")
                print("\n💡 Or specify a model path: python main.py --model path/to/model.h5")
                sys.exit(1)
        
        print(f"📂 Using model: {model_path}")
        print(f"📷 Using camera: {args.camera}")
        
        # Create and run the system
        system = EmotionRecognitionSystem(
            model_path=model_path,
            camera_id=args.camera
        )
        system.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Application terminated by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
