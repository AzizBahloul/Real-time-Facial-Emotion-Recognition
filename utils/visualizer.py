"""
Visualization Module
Handles professional UI rendering and real-time statistics display
"""

import cv2
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import deque
import time


class EmotionVisualizer:
    """Professional emotion visualization with real-time statistics"""
    
    def __init__(
        self,
        emotions: Dict[int, str],
        emotion_colors: Dict[str, Tuple[int, int, int]],
        history_length: int = 30
    ):
        """
        Initialize visualizer
        
        Args:
            emotions: Dictionary mapping emotion indices to names
            emotion_colors: Dictionary mapping emotion names to BGR colors
            history_length: Number of frames to keep for statistics
        """
        self.emotions = emotions
        self.emotion_colors = emotion_colors
        self.history_length = history_length
        
        # Statistics tracking
        self.emotion_history = deque(maxlen=history_length)
        self.confidence_history = deque(maxlen=history_length)
        self.fps_history = deque(maxlen=30)
        self.last_frame_time = time.time()
        
        # Session statistics
        self.session_emotions = {name: 0 for name in emotions.values()}
        self.total_detections = 0
        self.session_start_time = time.time()
    
    def update_stats(self, emotion: str, confidence: float):
        """Update statistics with new detection"""
        self.emotion_history.append(emotion)
        self.confidence_history.append(confidence)
        self.session_emotions[emotion] += 1
        self.total_detections += 1
    
    def calculate_fps(self) -> float:
        """Calculate current FPS"""
        current_time = time.time()
        fps = 1.0 / max(current_time - self.last_frame_time, 0.001)
        self.last_frame_time = current_time
        self.fps_history.append(fps)
        return np.mean(self.fps_history)
    
    def draw_face_box(
        self,
        frame: np.ndarray,
        bbox: Tuple[int, int, int, int],
        emotion: str,
        confidence: float
    ) -> np.ndarray:
        """
        Draw professional face bounding box with emotion label
        
        Args:
            frame: Input frame
            bbox: Face bounding box (x, y, w, h)
            emotion: Detected emotion
            confidence: Prediction confidence
            
        Returns:
            Frame with drawn elements
        """
        x, y, w, h = bbox
        color = self.emotion_colors.get(emotion, (255, 255, 255))
        
        # Draw rounded rectangle effect
        thickness = 2
        corner_length = 20
        
        # Top-left corner
        cv2.line(frame, (x, y), (x + corner_length, y), color, thickness)
        cv2.line(frame, (x, y), (x, y + corner_length), color, thickness)
        
        # Top-right corner
        cv2.line(frame, (x + w, y), (x + w - corner_length, y), color, thickness)
        cv2.line(frame, (x + w, y), (x + w, y + corner_length), color, thickness)
        
        # Bottom-left corner
        cv2.line(frame, (x, y + h), (x + corner_length, y + h), color, thickness)
        cv2.line(frame, (x, y + h), (x, y + h - corner_length), color, thickness)
        
        # Bottom-right corner
        cv2.line(frame, (x + w, y + h), (x + w - corner_length, y + h), color, thickness)
        cv2.line(frame, (x + w, y + h), (x + w, y + h - corner_length), color, thickness)
        
        # Draw emotion label with background
        label = f"{emotion}: {confidence:.1%}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        (label_w, label_h), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Label background
        label_y = y - 10 if y - 10 > label_h else y + h + label_h + 10
        cv2.rectangle(
            frame,
            (x, label_y - label_h - 5),
            (x + label_w + 10, label_y + 5),
            color,
            -1
        )
        
        # Label text
        cv2.putText(
            frame,
            label,
            (x + 5, label_y),
            font,
            font_scale,
            (255, 255, 255),
            font_thickness,
            cv2.LINE_AA
        )
        
        return frame
    
    def draw_confidence_bar(
        self,
        frame: np.ndarray,
        predictions: np.ndarray,
        x: int,
        y: int,
        width: int = 200,
        height: int = 150
    ) -> np.ndarray:
        """
        Draw confidence bars for all emotions
        
        Args:
            frame: Input frame
            predictions: Model predictions array
            x, y: Position for the bar chart
            width, height: Size of the chart area
            
        Returns:
            Frame with confidence bars
        """
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Title
        cv2.putText(
            frame,
            "Emotion Confidence",
            (x + 10, y + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Bars
        bar_height = 15
        bar_spacing = 18
        max_bar_width = width - 80
        
        for i, (idx, emotion) in enumerate(self.emotions.items()):
            conf = predictions[idx] if idx < len(predictions) else 0
            bar_y = y + 35 + i * bar_spacing
            bar_width = int(conf * max_bar_width)
            color = self.emotion_colors.get(emotion, (255, 255, 255))
            
            # Emotion label
            cv2.putText(
                frame,
                emotion[:3],
                (x + 5, bar_y + bar_height - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )
            
            # Bar background
            cv2.rectangle(
                frame,
                (x + 35, bar_y),
                (x + 35 + max_bar_width, bar_y + bar_height),
                (50, 50, 50),
                -1
            )
            
            # Bar fill
            if bar_width > 0:
                cv2.rectangle(
                    frame,
                    (x + 35, bar_y),
                    (x + 35 + bar_width, bar_y + bar_height),
                    color,
                    -1
                )
            
            # Percentage
            cv2.putText(
                frame,
                f"{conf:.0%}",
                (x + 40 + max_bar_width, bar_y + bar_height - 3),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.35,
                (255, 255, 255),
                1
            )
        
        return frame
    
    def draw_stats_panel(
        self,
        frame: np.ndarray,
        fps: float,
        faces_count: int
    ) -> np.ndarray:
        """
        Draw real-time statistics panel
        
        Args:
            frame: Input frame
            fps: Current FPS
            faces_count: Number of detected faces
            
        Returns:
            Frame with stats panel
        """
        h, w = frame.shape[:2]
        panel_width = 220
        panel_height = 200
        x = w - panel_width - 10
        y = 10
        
        # Panel background
        overlay = frame.copy()
        cv2.rectangle(overlay, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Panel border
        cv2.rectangle(frame, (x, y), (x + panel_width, y + panel_height), (100, 100, 100), 1)
        
        # Title
        cv2.putText(
            frame,
            "Real-Time Stats",
            (x + 10, y + 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )
        
        # Session duration
        duration = time.time() - self.session_start_time
        mins, secs = divmod(int(duration), 60)
        
        stats = [
            f"FPS: {fps:.1f}",
            f"Faces: {faces_count}",
            f"Duration: {mins:02d}:{secs:02d}",
            f"Total Detections: {self.total_detections}",
            "",
            "Session Summary:"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(
                frame,
                stat,
                (x + 10, y + 50 + i * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA
            )
        
        # Dominant emotion
        if self.total_detections > 0:
            dominant = max(self.session_emotions.items(), key=lambda x: x[1])
            dominant_pct = dominant[1] / self.total_detections * 100
            cv2.putText(
                frame,
                f"Dominant: {dominant[0]}",
                (x + 10, y + 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                self.emotion_colors.get(dominant[0], (255, 255, 255)),
                1,
                cv2.LINE_AA
            )
            cv2.putText(
                frame,
                f"({dominant_pct:.1f}%)",
                (x + 130, y + 168),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    def draw_instructions(self, frame: np.ndarray) -> np.ndarray:
        """Draw instruction overlay"""
        h, w = frame.shape[:2]
        
        instructions = [
            "Press 'Q' to quit",
            "Press 'S' to screenshot",
            "Press 'R' to reset stats"
        ]
        
        for i, instruction in enumerate(instructions):
            cv2.putText(
                frame,
                instruction,
                (10, h - 60 + i * 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (200, 200, 200),
                1,
                cv2.LINE_AA
            )
        
        return frame
    
    def reset_session_stats(self):
        """Reset session statistics"""
        self.session_emotions = {name: 0 for name in self.emotions.values()}
        self.total_detections = 0
        self.session_start_time = time.time()
        self.emotion_history.clear()
        self.confidence_history.clear()
