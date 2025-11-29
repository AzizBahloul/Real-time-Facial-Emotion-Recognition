"""
Face Detection Module
Handles face detection using OpenCV's Haar Cascade
"""

import cv2
import numpy as np
from typing import List, Tuple, Optional


class FaceDetector:
    """Face detection class using Haar Cascade Classifier"""
    
    def __init__(
        self,
        scale_factor: float = 1.3,
        min_neighbors: int = 5,
        min_face_size: Tuple[int, int] = (30, 30)
    ):
        """
        Initialize face detector
        
        Args:
            scale_factor: Parameter specifying how much the image size is reduced at each image scale
            min_neighbors: Parameter specifying how many neighbors each candidate rectangle should have
            min_face_size: Minimum possible face size
        """
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors
        self.min_face_size = min_face_size
        
        # Load Haar Cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        if self.face_cascade.empty():
            raise RuntimeError("Failed to load Haar Cascade classifier")
    
    def detect_faces(
        self,
        frame: np.ndarray,
        return_gray: bool = True
    ) -> Tuple[List[Tuple[int, int, int, int]], Optional[np.ndarray]]:
        """
        Detect faces in a frame
        
        Args:
            frame: Input BGR image
            return_gray: Whether to return grayscale image
            
        Returns:
            Tuple of (list of face bounding boxes, optional grayscale image)
        """
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=self.min_face_size,
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Convert to list of tuples
        face_list = [tuple(face) for face in faces]
        
        return (face_list, gray) if return_gray else (face_list, None)
    
    def extract_face_roi(
        self,
        gray_image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        target_size: Tuple[int, int] = (48, 48)
    ) -> np.ndarray:
        """
        Extract and preprocess face region of interest
        
        Args:
            gray_image: Grayscale image
            bbox: Face bounding box (x, y, w, h)
            target_size: Target size for the ROI
            
        Returns:
            Preprocessed face ROI ready for model input
        """
        x, y, w, h = bbox
        
        # Extract ROI
        roi = gray_image[y:y+h, x:x+w]
        
        # Resize to target size
        roi_resized = cv2.resize(roi, target_size, interpolation=cv2.INTER_AREA)
        
        return roi_resized
    
    def preprocess_for_model(
        self,
        roi: np.ndarray,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Preprocess ROI for model prediction
        
        Args:
            roi: Face ROI (grayscale)
            normalize: Whether to normalize pixel values
            
        Returns:
            Preprocessed array ready for model input
        """
        # Add batch and channel dimensions
        processed = np.expand_dims(roi, axis=0)
        processed = np.expand_dims(processed, axis=-1)
        
        # Normalize to [0, 1]
        if normalize:
            processed = processed.astype('float32') / 255.0
        
        return processed
