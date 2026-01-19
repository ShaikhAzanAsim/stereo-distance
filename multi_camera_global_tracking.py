"""
Multi-Camera Global ID + Distance Estimation Pipeline

A production-grade system for:
- Multi-camera object detection (YOLO)
- Global ID assignment across cameras (OSNet Re-ID)
- Stereo-based depth estimation
- Metric distance scaling using reference object
- Live visualization + encoded output

Author: Senior Computer Vision Engineer Shaikh Azan Asim
"""

import cv2
import torch
import numpy as np
from ultralytics import YOLO
import torchreid
import logging
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
from collections import defaultdict, deque
import time
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

# Video paths
TOP_LEFT_VIDEO = r""
TOP_RIGHT_VIDEO = r""
BOTTOM_VIDEO = r""

# Model paths
YOLO_MODEL_PATH = ""

# Reference object parameters
REF_OBJ_REAL_HEIGHT_CM = 8.0
REF_OBJ_DISTANCE_CM = 14.0
BASELINE_CM = 15.0

# Camera parameters
FOCAL_LENGTH_PX = 3060.0  # Hardcoded focal length for all cameras

# Matching parameters
COSINE_WEIGHT = 0.8  # Increased for better appearance matching across cameras
IOU_WEIGHT = 0.1     # Reduced since objects may appear in different positions
CLASS_WEIGHT = 0.1
MATCH_THRESHOLD = 0.5  # Lowered for more lenient cross-camera matching

# Display settings
DISPLAY_SCALE = 3  # Divide original size by this
DISPLAY_WINDOW_NAME = "Multi-Camera Global Tracking"

# Output settings
OUTPUT_VIDEO_PATH = "multi_cam_global_tracking_output.mp4"
OUTPUT_FPS = 30.0

# Performance settings
BATCH_SIZE = 6
MAX_TRACK_AGE = 60  # Increased to maintain IDs longer
EMBEDDING_HISTORY_SIZE = 15  # Increased for more stable embeddings

# Class filtering
SKIP_CLASSES = ['']  # Classes to skip during tracking and distance estimation

# Logging
LOG_LEVEL = logging.INFO

# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class Detection:
    """Single object detection."""
    bbox: np.ndarray  # [x1, y1, x2, y2]
    class_name: str
    confidence: float
    camera_id: int
    frame_number: int
    embedding: Optional[np.ndarray] = None
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bbox center."""
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def width(self) -> float:
        """Get bbox width."""
        return self.bbox[2] - self.bbox[0]
    
    @property
    def height(self) -> float:
        """Get bbox height."""
        return self.bbox[3] - self.bbox[1]


@dataclass
class Track:
    """Object track with history."""
    global_id: int
    class_name: str
    detections: deque = field(default_factory=lambda: deque(maxlen=EMBEDDING_HISTORY_SIZE))
    embeddings: deque = field(default_factory=lambda: deque(maxlen=EMBEDDING_HISTORY_SIZE))
    last_seen_frame: int = 0
    camera_ids: set = field(default_factory=set)
    
    def add_detection(self, detection: Detection):
        """Add detection to track history."""
        self.detections.append(detection)
        if detection.embedding is not None:
            self.embeddings.append(detection.embedding)
        self.last_seen_frame = detection.frame_number
        self.camera_ids.add(detection.camera_id)
    
    @property
    def avg_embedding(self) -> Optional[np.ndarray]:
        """Get average embedding from history."""
        if not self.embeddings:
            return None
        return np.mean(list(self.embeddings), axis=0)
    
    @property
    def last_detection(self) -> Optional[Detection]:
        """Get most recent detection."""
        return self.detections[-1] if self.detections else None


# ============================================================================
# LOGGING SETUP
# ============================================================================

def setup_logging():
    """Configure logging."""
    logging.basicConfig(
        level=LOG_LEVEL,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%H:%M:%S'
    )

logger = logging.getLogger(__name__)


# ============================================================================
# VIDEO STREAM HANDLER
# ============================================================================

class VideoStreamHandler:
    """Manages synchronized reading from 3 video streams."""
    
    def __init__(self, top_left_path: str, top_right_path: str, bottom_path: str):
        """
        Initialize video streams.
        
        Args:
            top_left_path: Path to top-left camera video
            top_right_path: Path to top-right camera video
            bottom_path: Path to bottom camera video
        """
        self.paths = [top_left_path, top_right_path, bottom_path]
        self.caps = []
        self.frame_count = 0
        
        # Open all streams
        for i, path in enumerate(self.paths):
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                raise ValueError(f"Failed to open video: {path}")
            self.caps.append(cap)
            logger.info(f"Opened camera {i}: {path}")
        
        # Get video properties from first stream
        self.width = int(self.caps[0].get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.caps[0].get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.caps[0].get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.caps[0].get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video properties: {self.width}x{self.height} @ {self.fps} FPS, {self.total_frames} frames")
    
    def read_frames(self) -> Optional[List[np.ndarray]]:
        """
        Read synchronized frames from all cameras.
        
        Returns:
            List of 3 frames [top_left, top_right, bottom] or None if end of video
        """
        frames = []
        
        for cap in self.caps:
            ret, frame = cap.read()
            if not ret:
                return None
            frames.append(frame)
        
        self.frame_count += 1
        return frames
    
    def release(self):
        """Release all video streams."""
        for cap in self.caps:
            cap.release()
        logger.info("Released all video streams")


# ============================================================================
# YOLO DETECTOR
# ============================================================================

class YOLODetector:
    """YOLO-based object detector."""
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        """
        Initialize YOLO detector.
        
        Args:
            model_path: Path to YOLO model weights
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading YOLO model from {model_path} on {self.device}")
        
        self.model = YOLO(model_path)
        self.model.to(self.device)
        
        logger.info(f"YOLO model loaded successfully")
    
    def detect(self, frames: List[np.ndarray], frame_number: int, 
               conf_threshold: float = 0.25) -> List[List[Detection]]:
        """
        Run detection on multiple frames.
        
        Args:
            frames: List of frames to process
            frame_number: Current frame number
            conf_threshold: Confidence threshold for detections
            
        Returns:
            List of detection lists, one per frame
        """
        all_detections = []
        
        # Batch inference
        with torch.no_grad():
            results = self.model(frames, verbose=False, conf=conf_threshold)
        
        # Parse results for each frame
        for camera_id, result in enumerate(results):
            detections = []
            
            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy().astype(int)
                
                for bbox, conf, cls_id in zip(boxes, confidences, class_ids):
                    class_name = result.names[cls_id]
                    
                    # Skip filtered classes
                    if class_name in SKIP_CLASSES:
                        continue
                    
                    detection = Detection(
                        bbox=bbox,
                        class_name=class_name,
                        confidence=float(conf),
                        camera_id=camera_id,
                        frame_number=frame_number
                    )
                    detections.append(detection)
            
            all_detections.append(detections)
            logger.debug(f"Camera {camera_id}: {len(detections)} detections")
        
        return all_detections


# ============================================================================
# OSNET RE-ID
# ============================================================================

class OSNetReID:
    """OSNet-based re-identification model."""
    
    def __init__(self, device: str = 'cuda'):
        """
        Initialize OSNet model.
        
        Args:
            device: Device to run inference on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        logger.info(f"Loading OSNet model on {self.device}")
        
        # Load pretrained OSNet
        self.model = torchreid.utils.FeatureExtractor(
            model_name='osnet_x1_0',
            model_path=None,  # Will download pretrained weights
            device=self.device
        )
        
        logger.info("OSNet model loaded successfully")
    
    def extract_embeddings(self, frames: List[np.ndarray], 
                          detections_list: List[List[Detection]]) -> List[List[Detection]]:
        """
        Extract embeddings for all detections.
        
        Args:
            frames: List of frames
            detections_list: List of detection lists
            
        Returns:
            Updated detection lists with embeddings
        """
        all_crops = []
        detection_indices = []
        
        # Collect all crops
        for frame_idx, (frame, detections) in enumerate(zip(frames, detections_list)):
            for det_idx, detection in enumerate(detections):
                x1, y1, x2, y2 = detection.bbox.astype(int)
                
                # Ensure valid crop
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                
                if x2 > x1 and y2 > y1:
                    crop = frame[y1:y2, x1:x2]
                    all_crops.append(crop)
                    detection_indices.append((frame_idx, det_idx))
        
        # Extract embeddings in batch
        if all_crops:
            with torch.no_grad():
                embeddings = self.model(all_crops)
                embeddings = embeddings.cpu().numpy()
            
            # Assign embeddings back to detections
            for (frame_idx, det_idx), embedding in zip(detection_indices, embeddings):
                # Normalize embedding
                embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
                detections_list[frame_idx][det_idx].embedding = embedding
        
        return detections_list


# ============================================================================
# GLOBAL ID MANAGER
# ============================================================================

class GlobalIDManager:
    """Manages global ID assignment across cameras."""
    
    def __init__(self):
        """Initialize global ID manager."""
        self.next_global_id = 1
        self.active_tracks: Dict[int, Track] = {}  # global_id -> Track
        self.camera_tracks: Dict[int, Dict[int, int]] = defaultdict(dict)  # camera_id -> {local_id -> global_id}
        
        logger.info("Global ID Manager initialized")
    
    def update(self, detections_list: List[List[Detection]], 
               frame_number: int) -> Dict[int, List[Tuple[Detection, int]]]:
        """
        Update tracks with new detections and assign global IDs.
        
        Args:
            detections_list: List of detection lists per camera
            frame_number: Current frame number
            
        Returns:
            Dict mapping camera_id to list of (detection, global_id) tuples
        """
        results = {}
        
        # Process each camera
        for camera_id, detections in enumerate(detections_list):
            camera_results = []
            
            for detection in detections:
                # Find best matching track
                best_match_id, best_score = self._find_best_match(detection, frame_number)
                
                if best_match_id is not None and best_score > MATCH_THRESHOLD:
                    # Update existing track
                    global_id = best_match_id
                    self.active_tracks[global_id].add_detection(detection)
                    logger.debug(f"Matched detection to global ID {global_id} (score: {best_score:.3f})")
                else:
                    # Create new track
                    global_id = self.next_global_id
                    self.next_global_id += 1
                    
                    track = Track(
                        global_id=global_id,
                        class_name=detection.class_name
                    )
                    track.add_detection(detection)
                    self.active_tracks[global_id] = track
                    
                    logger.debug(f"Created new global ID {global_id} for {detection.class_name}")
                
                camera_results.append((detection, global_id))
            
            results[camera_id] = camera_results
        
        # Clean up old tracks
        self._cleanup_old_tracks(frame_number)
        
        return results
    
    def _find_best_match(self, detection: Detection, 
                        frame_number: int) -> Tuple[Optional[int], float]:
        """
        Find best matching track for a detection.
        
        Args:
            detection: Detection to match
            frame_number: Current frame number
            
        Returns:
            Tuple of (global_id, match_score) or (None, 0.0)
        """
        best_id = None
        best_score = 0.0
        
        for global_id, track in self.active_tracks.items():
            # Skip if class doesn't match
            if track.class_name != detection.class_name:
                continue
            
            # Skip if track is too old
            if frame_number - track.last_seen_frame > MAX_TRACK_AGE:
                continue
            
            # Compute match score
            score = self._compute_match_score(detection, track)
            
            if score > best_score:
                best_score = score
                best_id = global_id
        
        return best_id, best_score
    
    def _compute_match_score(self, detection: Detection, track: Track) -> float:
        """
        Compute matching score between detection and track.
        
        Args:
            detection: Current detection
            track: Candidate track
            
        Returns:
            Match score in [0, 1]
        """
        score = 0.0
        
        # 1. Cosine similarity (80% - primary for cross-camera matching)
        if detection.embedding is not None and track.avg_embedding is not None:
            cosine_sim = np.dot(detection.embedding, track.avg_embedding)
            cosine_sim = np.clip(cosine_sim, 0, 1)
            score += COSINE_WEIGHT * cosine_sim
        
        # 2. Temporal IoU (10% - only for same camera)
        last_det = track.last_detection
        if last_det is not None and last_det.camera_id == detection.camera_id:
            iou = self._compute_iou(detection.bbox, last_det.bbox)
            score += IOU_WEIGHT * iou
        
        # 3. Class consistency (10%)
        if detection.class_name == track.class_name:
            score += CLASS_WEIGHT * 1.0
        
        # Bonus: Cross-camera consistency
        # If this track has been seen in multiple cameras, give it a small boost
        # This helps maintain the same ID across all cameras
        if len(track.camera_ids) > 1:
            score += 0.05  # Small bonus for multi-camera tracks
        
        return score
    
    @staticmethod
    def _compute_iou(bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Compute IoU between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / (union + 1e-8)
    
    def _cleanup_old_tracks(self, frame_number: int):
        """Remove tracks that haven't been seen recently."""
        to_remove = []
        
        for global_id, track in self.active_tracks.items():
            if frame_number - track.last_seen_frame > MAX_TRACK_AGE:
                to_remove.append(global_id)
        
        for global_id in to_remove:
            del self.active_tracks[global_id]
            logger.debug(f"Removed old track {global_id}")


# ============================================================================
# STEREO DEPTH ESTIMATOR
# ============================================================================

class StereoDepthEstimator:
    """Hybrid depth estimation: stereo disparity + reference object scaling."""
    
    def __init__(self, baseline_cm: float, ref_height_cm: float, ref_distance_cm: float):
        """
        Initialize depth estimator.
        
        Args:
            baseline_cm: Stereo baseline in cm
            ref_height_cm: Reference object height in cm
            ref_distance_cm: Reference object distance in cm
        """
        self.baseline_cm = baseline_cm
        self.ref_height_cm = ref_height_cm
        self.ref_distance_cm = ref_distance_cm
        self.focal_length_px = FOCAL_LENGTH_PX  # Use hardcoded value
        self.calibrated = True  # Always calibrated with hardcoded focal length
        
        # For monocular depth estimation
        self.ref_pixel_height = None  # Will be set when ref-obj is detected
        
        logger.info(f"Depth estimator initialized (baseline={baseline_cm}cm, focal_length={FOCAL_LENGTH_PX}px)")
    
    def calibrate(self, bottom_detections: List[Detection], frame: np.ndarray) -> Optional[Tuple[float, int]]:
        """
        Find reference object to set ref_pixel_height for monocular estimation.
        
        Args:
            bottom_detections: Detections from bottom camera
            frame: Bottom camera frame
            
        Returns:
            Tuple of (ref_distance_cm, ref_global_id) or None
        """
        # Find ref-obj detection
        ref_obj = None
        for det in bottom_detections:
            if 'ref' in det.class_name.lower():
                ref_obj = det
                break
        
        if ref_obj is None:
            return None
        
        # Store reference pixel height for monocular estimation
        pixel_height = ref_obj.height
        self.ref_pixel_height = pixel_height
        
        logger.info(f"Reference object detected: height={pixel_height:.1f}px (focal_length={self.focal_length_px:.1f}px)")
        
        # Return reference object info
        return (self.ref_distance_cm, None)
    
    def estimate_depth_stereo(self, left_results: List[Tuple[Detection, int]], 
                             right_results: List[Tuple[Detection, int]]) -> Dict[int, float]:
        """
        Estimate depth for objects visible in both top cameras using stereo disparity.
        
        Args:
            left_results: (detection, global_id) tuples from left camera
            right_results: (detection, global_id) tuples from right camera
            
        Returns:
            Dict mapping global_id to depth in cm
        """
        if not self.calibrated:
            return {}
        
        depths = {}
        
        # Build lookup for right camera detections
        right_lookup = {global_id: det for det, global_id in right_results}
        
        # Count potential matches
        left_ids = {gid for _, gid in left_results}
        right_ids = {gid for _, gid in right_results}
        common_ids = left_ids & right_ids
        
        if common_ids:
            logger.debug(f"Found {len(common_ids)} objects in both cameras: {common_ids}")
        
        # Match left detections with right
        for left_det, global_id in left_results:
            if global_id not in right_lookup:
                continue
            
            right_det = right_lookup[global_id]
            
            # Compute disparity (x_left - x_right)
            left_x = left_det.center[0]
            right_x = right_det.center[0]
            disparity = left_x - right_x
            
            # Validate disparity
            if disparity <= 0:
                logger.debug(f"Invalid disparity for ID {global_id} ({left_det.class_name}): {disparity:.2f} (left_x={left_x:.1f}, right_x={right_x:.1f})")
                continue
            
            # Compute depth: Z = (f * B) / disparity
            depth_cm = (self.focal_length_px * self.baseline_cm) / disparity
            
            # Sanity check
            if 10 < depth_cm < 1000:  # Reasonable range
                depths[global_id] = depth_cm
                logger.info(f"✓ [STEREO] ID {global_id} ({left_det.class_name}): distance = {depth_cm:.1f} cm (disparity = {disparity:.2f}px)")
            else:
                logger.debug(f"Out of range depth for ID {global_id}: {depth_cm:.1f} cm")
        
        return depths
    
    def estimate_depth_monocular(self, all_results: Dict[int, List[Tuple[Detection, int]]]) -> Dict[int, Dict[str, float]]:
        """
        Estimate depth using monocular cues (reference object scaling).
        Uses the principle: distance = (real_height × focal_length) / pixel_height
        
        Args:
            all_results: Detection results per camera
            
        Returns:
            Dict mapping global_id to dict of {camera_id: depth_cm}
        """
        if not self.calibrated:
            return {}
        
        mono_depths = {}
        
        # Process all cameras
        for camera_id, results in all_results.items():
            for detection, global_id in results:
                # Skip reference object itself
                if 'ref' in detection.class_name.lower():
                    continue
                
                # Get pixel height of detection
                pixel_height = detection.height
                
                # Estimate distance using similar triangles
                # If ref object has height H_real at distance D with pixel height P_ref,
                # then an object with same real height at distance D' will have pixel height P'
                # D' = D × (P_ref / P')
                
                # For now, we assume objects have similar height to reference object
                # This is a simplification - in production, you'd have known heights per class
                estimated_distance = self.ref_distance_cm * (self.ref_pixel_height / pixel_height)
                
                # Sanity check
                if 10 < estimated_distance < 1000:
                    if global_id not in mono_depths:
                        mono_depths[global_id] = {}
                    mono_depths[global_id][camera_id] = estimated_distance
                    
                    logger.debug(f"[MONO] ID {global_id} ({detection.class_name}) cam{camera_id}: ~{estimated_distance:.1f}cm (height={pixel_height:.1f}px)")
        
        return mono_depths
    
    def estimate_depth_hybrid(self, left_results: List[Tuple[Detection, int]], 
                             right_results: List[Tuple[Detection, int]],
                             all_results: Dict[int, List[Tuple[Detection, int]]],
                             bottom_results: List[Tuple[Detection, int]]) -> Tuple[Dict[int, Tuple[float, str]], Optional[Tuple[int, float]]]:
        """
        Hybrid depth estimation: prefer stereo, fallback to monocular.
        Also calculates reference object distance in real-time.
        
        Args:
            left_results: Top-left camera results
            right_results: Top-right camera results
            all_results: All camera results
            bottom_results: Bottom camera results (for ref-obj)
            
        Returns:
            Tuple of:
            - Dict mapping global_id to (depth_cm, method) where method is 'stereo' or 'mono'
            - Optional tuple of (ref_global_id, ref_distance_cm) for reference object
        """
        final_depths = {}
        ref_obj_info = None
        
        # Calculate reference object distance in real-time
        if self.calibrated:
            for detection, global_id in bottom_results:
                if 'ref' in detection.class_name.lower():
                    # Calculate distance using: distance = (focal_length × real_height) / pixel_height
                    pixel_height = detection.height
                    ref_distance = (self.focal_length_px * self.ref_height_cm) / pixel_height
                    ref_obj_info = (global_id, ref_distance)
                    logger.debug(f"[REF-OBJ] ID {global_id}: distance = {ref_distance:.1f} cm (height={pixel_height:.1f}px)")
                    break
        
        # 1. Get stereo depths (most accurate)
        stereo_depths = self.estimate_depth_stereo(left_results, right_results)
        for global_id, depth in stereo_depths.items():
            final_depths[global_id] = (depth, 'stereo')
        
        # 2. Get monocular depths for objects not in stereo
        mono_depths = self.estimate_depth_monocular(all_results)
        
        for global_id, camera_depths in mono_depths.items():
            if global_id in final_depths:
                # Already have stereo depth, skip
                continue
            
            # Average depths from multiple cameras if available
            avg_depth = np.mean(list(camera_depths.values()))
            final_depths[global_id] = (avg_depth, 'mono')
            
            # Get class name for logging
            for camera_id, results in all_results.items():
                for det, gid in results:
                    if gid == global_id:
                        logger.info(f"✓ [MONO] ID {global_id} ({det.class_name}): distance = {avg_depth:.1f} cm (ref-based)")
                        break
                break
        
        return final_depths, ref_obj_info



# ============================================================================
# VISUALIZER
# ============================================================================

class Visualizer:
    """Multi-camera visualization with overlays."""
    
    def __init__(self, frame_width: int, frame_height: int, scale: int = 3):
        """
        Initialize visualizer.
        
        Args:
            frame_width: Original frame width
            frame_height: Original frame height
            scale: Downscale factor for display
        """
        self.display_width = frame_width // scale
        self.display_height = frame_height // scale
        self.scale = scale
        
        # Combined display size (3 cameras side-by-side)
        self.combined_width = self.display_width * 3
        self.combined_height = self.display_height
        
        # Colors for different classes
        self.class_colors = {
            'vessel-ship': (0, 255, 255),      # Yellow
            'vessel-jetski': (255, 0, 255),    # Magenta
            'person': (0, 255, 0),             # Green
            'ref-obj': (0, 0, 255),            # Red
        }
        
        logger.info(f"Visualizer initialized: {self.combined_width}x{self.combined_height}")
    
    def draw_results(self, frames: List[np.ndarray], 
                    all_results: Dict[int, List[Tuple[Detection, int]]],
                    depths: Dict[int, Tuple[float, str]],
                    ref_obj_info: Optional[Tuple[int, float]],
                    fps: float) -> np.ndarray:
        """
        Draw visualization with all overlays.
        
        Args:
            frames: List of 3 frames
            all_results: Detection results per camera
            depths: Depth estimates per global_id as (depth_cm, method)
            ref_obj_info: Optional tuple of (ref_global_id, ref_distance_cm)
            fps: Current FPS
            
        Returns:
            Combined visualization frame
        """
        vis_frames = []
        
        for camera_id, frame in enumerate(frames):
            # Resize frame
            vis_frame = cv2.resize(frame, (self.display_width, self.display_height))
            
            # Get results for this camera
            results = all_results.get(camera_id, [])
            
            # Draw each detection
            for detection, global_id in results:
                # Scale bbox
                bbox = detection.bbox / self.scale
                x1, y1, x2, y2 = bbox.astype(int)
                
                # Get color
                color = self.class_colors.get(detection.class_name, (255, 255, 255))
                
                # Draw bbox
                cv2.rectangle(vis_frame, (x1, y1), (x2, y2), color, 2)
                
                # Prepare labels (two lines: ID+class, then distance)
                line1 = f"ID:{global_id} | {detection.class_name}"
                line2 = ""
                
                # Add distance (hybrid: stereo or monocular)
                if global_id in depths:
                    distance_cm, method = depths[global_id]
                    method_tag = "S" if method == "stereo" else "M"
                    line2 = f"{distance_cm:.1f}cm [{method_tag}]"
                # Add calculated distance for reference object
                elif ref_obj_info and global_id == ref_obj_info[0]:
                    ref_distance = ref_obj_info[1]
                    line2 = f"{ref_distance:.1f}cm [calc]"
                
                # Calculate label dimensions
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1
                (w1, h1), _ = cv2.getTextSize(line1, font, font_scale, thickness)
                
                # Draw line 1 (ID + class)
                y_offset = y1 - 5
                cv2.rectangle(vis_frame, (x1, y_offset - h1 - 3), (x1 + w1, y_offset), color, -1)
                cv2.putText(vis_frame, line1, (x1, y_offset - 3), font, font_scale, (0, 0, 0), thickness)
                
                # Draw line 2 (distance) if available
                if line2:
                    (w2, h2), _ = cv2.getTextSize(line2, font, font_scale, thickness)
                    y_offset2 = y_offset - h1 - 5
                    cv2.rectangle(vis_frame, (x1, y_offset2 - h2 - 3), (x1 + w2, y_offset2), color, -1)
                    cv2.putText(vis_frame, line2, (x1, y_offset2 - 3), font, font_scale, (0, 0, 0), thickness)
            
            # Add camera label
            cam_labels = ["TOP LEFT", "TOP RIGHT", "BOTTOM"]
            cv2.putText(vis_frame, cam_labels[camera_id], (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            vis_frames.append(vis_frame)
        
        # Combine frames horizontally
        combined = np.hstack(vis_frames)
        
        # Draw FPS
        cv2.putText(combined, f"FPS: {fps:.1f}", (10, self.combined_height - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return combined


# ============================================================================
# MAIN PIPELINE
# ============================================================================

class MainPipeline:
    """Main processing pipeline orchestrator."""
    
    def __init__(self):
        """Initialize pipeline components."""
        logger.info("Initializing Multi-Camera Global Tracking Pipeline")
        
        # Initialize components
        self.stream_handler = VideoStreamHandler(TOP_LEFT_VIDEO, TOP_RIGHT_VIDEO, BOTTOM_VIDEO)
        self.detector = YOLODetector(YOLO_MODEL_PATH)
        self.reid = OSNetReID()
        self.id_manager = GlobalIDManager()
        self.depth_estimator = StereoDepthEstimator(BASELINE_CM, REF_OBJ_REAL_HEIGHT_CM, REF_OBJ_DISTANCE_CM)
        self.visualizer = Visualizer(self.stream_handler.width, self.stream_handler.height, DISPLAY_SCALE)
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(
            OUTPUT_VIDEO_PATH,
            fourcc,
            OUTPUT_FPS,
            (self.visualizer.combined_width, self.visualizer.combined_height)
        )
        
        # Depth tracking for plotting
        self.depth_history = defaultdict(lambda: {'frames': [], 'depths': [], 'class': None})
        
        logger.info("Pipeline initialized successfully")
    
    def run(self):
        """Run main processing loop."""
        logger.info("Starting processing...")
        
        frame_times = deque(maxlen=30)
        
        try:
            while True:
                start_time = time.time()
                
                # Read frames
                frames = self.stream_handler.read_frames()
                if frames is None:
                    logger.info("End of video reached")
                    break
                
                frame_number = self.stream_handler.frame_count
                
                # Run detection
                detections_list = self.detector.detect(frames, frame_number)
                
                # Extract embeddings
                detections_list = self.reid.extract_embeddings(frames, detections_list)
                
                # Detect reference object (for monocular estimation baseline)
                if self.depth_estimator.ref_pixel_height is None:
                    self.depth_estimator.calibrate(detections_list[2], frames[2])
                
                # Assign global IDs
                all_results = self.id_manager.update(detections_list, frame_number)
                
                # Estimate depth using hybrid approach (stereo + monocular)
                # Also calculates reference object distance in real-time
                depths, ref_obj_info = self.depth_estimator.estimate_depth_hybrid(
                    all_results.get(0, []),  # left
                    all_results.get(1, []),  # right
                    all_results,             # all cameras for monocular
                    all_results.get(2, [])   # bottom camera for ref-obj
                )
                
                # Track depth data for plotting
                for global_id, (depth_cm, method) in depths.items():
                    # Find class name for this global_id
                    for camera_id, results in all_results.items():
                        for det, gid in results:
                            if gid == global_id:
                                self.depth_history[global_id]['frames'].append(frame_number)
                                self.depth_history[global_id]['depths'].append(depth_cm)
                                self.depth_history[global_id]['class'] = det.class_name
                                break
                        if self.depth_history[global_id]['class'] is not None:
                            break
                
                # Track reference object if available
                if ref_obj_info:
                    ref_id, ref_distance = ref_obj_info
                    self.depth_history[ref_id]['frames'].append(frame_number)
                    self.depth_history[ref_id]['depths'].append(ref_distance)
                    self.depth_history[ref_id]['class'] = 'ref-obj'
                
                # Visualize
                elapsed = time.time() - start_time
                frame_times.append(elapsed)
                fps = 1.0 / (np.mean(frame_times) + 1e-8)
                
                vis_frame = self.visualizer.draw_results(frames, all_results, depths, ref_obj_info, fps)
                
                # Display
                cv2.imshow(DISPLAY_WINDOW_NAME, vis_frame)
                
                # Write to output
                self.video_writer.write(vis_frame)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("User requested quit")
                    break
                
                # Progress logging
                if frame_number % 30 == 0:
                    logger.info(f"Frame {frame_number}/{self.stream_handler.total_frames} "
                              f"({100*frame_number/self.stream_handler.total_frames:.1f}%) - "
                              f"FPS: {fps:.1f}")
        
        except KeyboardInterrupt:
            logger.info("Interrupted by user")
        
        except Exception as e:
            logger.error(f"Error during processing: {e}", exc_info=True)
        
        finally:
            self.cleanup()
    
    def plot_depth_graph(self):
        """Generate and save depth vs frame number plot."""
        if not self.depth_history:
            logger.warning("No depth data to plot")
            return
        
        logger.info("Generating depth plot...")
        
        # Group data by class
        class_data = defaultdict(lambda: {'frames': [], 'depths': []})
        
        for global_id, data in self.depth_history.items():
            if data['class'] and data['frames']:
                class_name = data['class']
                class_data[class_name]['frames'].extend(data['frames'])
                class_data[class_name]['depths'].extend(data['depths'])
        
        # Create plot
        plt.figure(figsize=(14, 8))
        
        # Plot each class
        colors = {
            'vessel-jetski': 'blue',
            'vessel-ship': 'green',
            'ref-obj': 'red',
            'person': 'orange'
        }
        
        for class_name, data in class_data.items():
            if data['frames']:
                color = colors.get(class_name, 'gray')
                plt.plot(data['frames'], data['depths'], 
                        marker='o', markersize=2, linestyle='-', linewidth=1,
                        color=color, label=class_name, alpha=0.7)
        
        plt.xlabel('Frame Number', fontsize=12)
        plt.ylabel('Distance (cm)', fontsize=12)
        plt.title('Object Distance vs Frame Number', fontsize=14, fontweight='bold')
        plt.legend(loc='best', fontsize=10)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save plot
        plot_path = 'depth_plot.png'
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"Depth plot saved to: {plot_path}")
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up...")
        
        self.stream_handler.release()
        self.video_writer.release()
        cv2.destroyAllWindows()
        
        # Generate depth plot
        self.plot_depth_graph()
        
        logger.info(f"Output saved to: {OUTPUT_VIDEO_PATH}")
        logger.info("Pipeline finished")


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    """Main entry point."""
    setup_logging()
    
    logger.info("=" * 80)
    logger.info("Multi-Camera Global ID + Distance Estimation Pipeline")
    logger.info("=" * 80)
    
    # Run pipeline
    pipeline = MainPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()
