"""
Data Loading Module
Handles dataset downloading, loading, and preprocessing
"""

import os
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import warnings
warnings.filterwarnings("ignore")


class DatasetLoader:
    """Dataset loader for FER2013 and similar emotion datasets"""
    
    def __init__(self, data_dir: Optional[Path] = None):
        """
        Initialize data loader
        
        Args:
            data_dir: Directory to store/load data from
        """
        self.data_dir = data_dir
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']
        self.num_classes = len(self.emotions)
        self.image_size = (48, 48)
    
    @staticmethod
    def download_fer2013() -> str:
        """
        Download FER2013 dataset using kagglehub
        
        Returns:
            Path to downloaded dataset
        """
        try:
            import kagglehub
            path = kagglehub.dataset_download("msambare/fer2013")
            print(f"Dataset downloaded to: {path}")
            return path
        except ImportError:
            raise ImportError(
                "kagglehub not installed. Install with: pip install kagglehub"
            )
        except Exception as e:
            raise RuntimeError(f"Failed to download dataset: {e}")
    
    def load_images_from_directory(
        self,
        directory: str,
        target_size: Tuple[int, int] = (48, 48)
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load images from directory structure (FER2013 format)
        
        Args:
            directory: Root directory containing emotion subdirectories
            target_size: Target image size
            
        Returns:
            Tuple of (images array, labels array)
        """
        import cv2
        
        images = []
        labels = []
        
        directory = Path(directory)
        
        for emotion_idx, emotion in enumerate(self.emotions):
            emotion_dir = directory / emotion
            
            if not emotion_dir.exists():
                print(f"Warning: Directory not found: {emotion_dir}")
                continue
            
            for img_path in emotion_dir.glob("*.jpg"):
                try:
                    # Load as grayscale
                    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                    
                    if img is None:
                        continue
                    
                    # Resize if needed
                    if img.shape != target_size:
                        img = cv2.resize(img, target_size, interpolation=cv2.INTER_AREA)
                    
                    images.append(img)
                    labels.append(emotion_idx)
                    
                except Exception as e:
                    print(f"Error loading {img_path}: {e}")
                    continue
            
            print(f"Loaded {emotion}: {labels.count(emotion_idx)} images")
        
        # Convert to numpy arrays
        images = np.array(images)
        labels = np.array(labels)
        
        # Reshape and normalize
        images = images.reshape(-1, target_size[0], target_size[1], 1)
        images = images.astype('float32') / 255.0
        
        return images, labels
    
    def load_dataset(
        self,
        train_dir: str,
        test_dir: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load complete dataset with train and test splits
        
        Args:
            train_dir: Directory containing training images
            test_dir: Directory containing test images
            
        Returns:
            Tuple of (X_train, y_train, X_test, y_test)
        """
        print("Loading training data...")
        X_train, y_train = self.load_images_from_directory(train_dir)
        
        print("\nLoading test data...")
        X_test, y_test = self.load_images_from_directory(test_dir)
        
        print(f"\nDataset loaded:")
        print(f"  Training samples: {len(X_train)}")
        print(f"  Test samples: {len(X_test)}")
        
        return X_train, y_train, X_test, y_test
    
    def get_class_weights(self, labels: np.ndarray) -> dict:
        """
        Calculate class weights for imbalanced dataset
        
        Args:
            labels: Label array
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(labels)
        weights = compute_class_weight(
            class_weight='balanced',
            classes=classes,
            y=labels
        )
        
        return dict(zip(classes, weights))
    
    def get_dataset_stats(
        self,
        labels: np.ndarray
    ) -> dict:
        """
        Get dataset statistics
        
        Args:
            labels: Label array
            
        Returns:
            Dictionary with statistics
        """
        unique, counts = np.unique(labels, return_counts=True)
        
        stats = {
            'total_samples': len(labels),
            'num_classes': len(unique),
            'class_distribution': {
                self.emotions[i]: int(count) 
                for i, count in zip(unique, counts)
            },
            'class_balance': {
                self.emotions[i]: count / len(labels) * 100
                for i, count in zip(unique, counts)
            }
        }
        
        return stats
