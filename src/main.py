#!/usr/bin/env python3
"""
Live ANPR Detection System
Supports both camera and video input with GUI interface
Uses YOLO for vehicle/plate detection and custom CRNN for OCR
"""

import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image, ImageFont, ImageDraw
import os
from pathlib import Path
import time
import logging
from datetime import datetime
from collections import deque, defaultdict
import threading
import queue
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict
import re

# PyQt5 imports for GUI
from PyQt5.QtWidgets import (QApplication, QMainWindow, QVBoxLayout, QHBoxLayout, 
                            QWidget, QLabel, QPushButton, QComboBox, QFileDialog,
                            QTextEdit, QGroupBox, QProgressBar, QCheckBox, QSpinBox,
                            QSlider, QFrame, QSplitter)
from PyQt5.QtCore import QThread, pyqtSignal, QTimer, Qt
from PyQt5.QtGui import QImage, QPixmap, QFont

# Add beam search decoding
try:
    from fast_ctc_decode import beam_search
    BEAM_SEARCH_AVAILABLE = True
except ImportError:
    BEAM_SEARCH_AVAILABLE = False
    logging.warning("fast_ctc_decode not available. Using greedy decoding.")

# YOLO import
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    print("Warning: ultralytics not available. YOLO detection will be disabled.")
    YOLO_AVAILABLE = False

# Setup logging
log_dir = Path("logs")
log_dir.mkdir(exist_ok=True)
log_file = log_dir / f"anpr_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,  # Changed to INFO to reduce spam but keep important messages
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
class Config:
    # Get the project root directory (parent of src/)
    PROJECT_ROOT = Path(__file__).resolve().parent.parent
    
    # Model paths - Using relative paths for cross-platform compatibility
    # Models are in: models/application_runner/
    YOLO_MODEL_PATH = PROJECT_ROOT / "models" / "application_runner" / "best.pt"
    CRNN_MODEL_PATH = PROJECT_ROOT / "models" / "application_runner" / "best_crnn_model_epoch125_acc0.9010.pth"
    
    # Model Information
    YOLO_MODEL_NAME = "yolo11n_anpr"  # YOLO11n for ANPR detection
    CRNN_MODEL_VERSION = "v7"  # CRNN version 7 (90.10% accuracy)
    
    # Video path
    VIDEO_PATH = PROJECT_ROOT / "data" / "Input"
    
    # Camera configuration
    DEFAULT_CAMERA_INDEX = 0
    CAMERA_WIDTH = 1280  # Default camera resolution
    CAMERA_HEIGHT = 720
    CAMERA_FPS = 30
    
    # Detection parameters
    CONFIDENCE_THRESHOLD = 0.3  # Reasonable threshold for better detection
    NMS_THRESHOLD = 0.45
    MIN_PLATE_AREA = 400  # Minimum area for license plate (increased to reduce false positives)
    PLATE_CONFIDENCE_THRESHOLD = 0.2  # Separate threshold for plates
    
    # OCR parameters  
    OCR_IMG_HEIGHT = 64
    OCR_IMG_WIDTH = 256
    MIN_OCR_CONFIDENCE = 0.1  # Temporarily lowered from 0.3 for debugging
    
    # Duplicate prevention
    DUPLICATE_TIME_WINDOW = 5.0  # seconds
    SIMILARITY_THRESHOLD = 0.8
    
    # Output
    OUTPUT_DIR = PROJECT_ROOT / "outputs"
    SAVE_DETECTIONS = True # Default to saving detections
    NUM_VERTICAL_ROWS = 3 # For multi-line CRNN model

@dataclass
class Detection:
    """Class to store detection information"""
    text: str
    confidence: float
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    timestamp: float
    frame_id: int
    detection_type: str = "plate"  # "car" or "plate"
    plate_type: str = "unknown"  # "green", "white", "red", "unknown"
    
class ImprovedBidirectionalLSTM(nn.Module):
    """Improved Bidirectional LSTM layer"""
    def __init__(self, input_size, hidden_size, output_size, num_layers=2, dropout=0.1):
        super(ImprovedBidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(
            input_size, 
            hidden_size, 
            num_layers=num_layers,
            bidirectional=True, 
            batch_first=False,
            dropout=dropout if num_layers > 1 else 0
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)
        self.dropout_layer = nn.Dropout(dropout)

    def forward(self, input_tensor):
        recurrent, _ = self.rnn(input_tensor)
        output = self.dropout_layer(recurrent)
        output = self.linear(output)
        return output

class CustomCRNN(nn.Module):
    """Custom CRNN model for license plate recognition (matching train_custom_crnn.py exactly)"""
    def __init__(self, img_height, n_classes, n_hidden=256):
        super(CustomCRNN, self).__init__()
        
        # CNN part exactly as in training script
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(32), nn.ReLU(True), nn.Dropout2d(0.3 * 0.5),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 64x32x128

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True), nn.Dropout2d(0.3 * 0.5),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2), # 128x16x64

            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True), nn.Dropout2d(0.3 * 0.7),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 256x8x64

            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)), # 512x4x64
            
            nn.Conv2d(512, 512, kernel_size=2, stride=1, padding=0), nn.BatchNorm2d(512), nn.ReLU(True), nn.Dropout2d(0.3) # 512x3x63
        )
        
        # RNN part exactly as in training script
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, None)) # Output H will be 1
        
        # Calculate RNN input size based on CNN output - exactly as in training script
        self.rnn_input_size = 512 
        self.rnn1 = ImprovedBidirectionalLSTM(self.rnn_input_size, n_hidden // 2, n_hidden // 2, num_layers=2, dropout=0.3)
        self.rnn2 = ImprovedBidirectionalLSTM(n_hidden // 2, n_hidden // 2, n_classes, num_layers=1, dropout=0.3)
        
    def forward(self, input_tensor):
        # Forward logic exactly as in training script
        conv = self.cnn(input_tensor)
        conv = self.adaptive_pool(conv)
        conv = conv.squeeze(2) # Remove height dimension (H=1): (batch, channels, width)
        conv = conv.permute(2, 0, 1) # (width, batch, channels) for RNN
        
        output = self.rnn1(conv)
        output = self.rnn2(output) # Output shape: (seq_len, batch, num_classes)
        return output

class ANPRProcessor:
    """Main ANPR processing class"""
    
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"Using device: {self.device}")
        
        # Initialize models
        self.yolo_model = None
        self.crnn_model = None
        self.char_list = None
        self.yolo_loaded = False
        self.crnn_loaded = False
        
        # Class indices for YOLO model - will be loaded dynamically
        self.plate_class_idx = -1
        self.vehicle_class_indices = []
        
        # Detection tracking
        self.recent_detections = deque(maxlen=100)
        self.detection_history = defaultdict(list)
        self.processed_cars = {}  # Track processed cars to avoid reprocessing
        
        # Unique plate tracking (avoid continuous sequences)
        self.last_detected_plates = {}  # plate_text -> timestamp
        self.unique_detection_window = 10.0  # seconds - minimum time between same plate detections
        
        # Car image saving tracking (one image per car)
        self.saved_car_plates = set()  # Track which plates have been saved as car images
        
        # Parking vehicle detection tracking
        self.car_tracking = {}  # Track car positions over time for parking detection
        self.parked_cars = {}  # Cars that have been identified as parked
        self.parking_detection_enabled = False
        self.parking_time_threshold = 3.0  # seconds
        
        # Vehicle-plate association filtering
        self.vehicle_plate_association_enabled = False
        
        # Zone crossing state tracking
        self.zone_vehicle_states = {}  # vehicle_id -> {"state": "entered", "last_seen": timestamp, "processed": False}
        self.tracked_vehicles_zone = {}  # vehicle_id -> last_position (center_x, center_y)
        self.zone_processed_plates = set()  # Track plates already processed in current zone session
        
        # Frame skipping for real-time processing
        self.frame_skip_counter = 0
        self.process_every_n_frames = 3  # Process every 3rd frame for real-time performance
        
        # Stats
        self.total_detections = 0
        self.unique_plates = set()
        
        # Plate type statistics
        self.plate_type_counts = {
            'white': 0,
            'green': 0,
            'red': 0,
            'yellow': 0,
            'blue': 0,
            'unknown': 0
        }
        
        Config.OUTPUT_DIR.mkdir(exist_ok=True)
        
    def load_models(self):
        """Load YOLO and CRNN models"""
        try:
            # Load YOLO model
            if YOLO_AVAILABLE and Path(Config.YOLO_MODEL_PATH).exists():
                logger.info(f"Loading YOLO model from: {Config.YOLO_MODEL_PATH}")
                self.yolo_model = YOLO(Config.YOLO_MODEL_PATH)
                # Force CPU to avoid CUDA torchvision NMS issues
                self.yolo_model.to('cpu')
                logger.info("YOLO model loaded successfully (CPU mode)")
                self.yolo_loaded = True

                # Dynamically determine class indices from the model
                if hasattr(self.yolo_model, 'names'):
                    class_names = self.yolo_model.names
                    self.vehicle_class_indices = [k for k, v in class_names.items() if v.lower() in ['car', 'motorcycle', 'vehicle']]
                    plate_indices = [k for k, v in class_names.items() if 'plate' in v.lower()]
                    if plate_indices:
                        self.plate_class_idx = plate_indices[0]
                    
                    logger.info(f"Dynamically loaded YOLO class indices -> Vehicles: {self.vehicle_class_indices}, Plate: {self.plate_class_idx}")
                else:
                    # Fallback to hardcoded indices if names attribute is not available
                    logger.warning("Could not read class names from YOLO model. Using hardcoded indices.")
                    self.vehicle_class_indices = [0, 1]
                    self.plate_class_idx = 2
            else:
                logger.warning("YOLO model not available")
                self.yolo_loaded = False
                
            # Load CRNN model
            self._load_crnn_model()
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            return False
    
    def _load_crnn_model(self):
        """Load CRNN model for OCR"""
        try:
            model_path = Config.CRNN_MODEL_PATH
                
            if not Path(model_path).exists():
                raise FileNotFoundError(f"CRNN model not found at: {model_path}")
                
            logger.info(f"Loading CRNN model from: {model_path}")
            
            # Load model checkpoint
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get character set
            if 'char_set' in checkpoint:
                self.char_list = checkpoint['char_set']
            else:
                # Fallback character set
                logger.warning("Character set not found, using fallback")
                self.char_list = ['[blank]'] + list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
            
            # Get model config
            model_config = checkpoint.get('model_config', {})
            img_height = model_config.get('img_height', Config.OCR_IMG_HEIGHT)
            n_classes = model_config.get('n_classes', len(self.char_list))
            n_hidden = model_config.get('n_hidden', 256)
            
            logger.info(f"Model config: img_height={img_height}, n_classes={n_classes}, n_hidden={n_hidden}")
            
            # The new model from training script is always 3-channel RGB
            input_channels = 3
            logger.info(f"Using RGB model with {input_channels} input channels")
            
            # Initialize model with the exact same architecture as training script
            self.crnn_model = CustomCRNN(img_height, n_classes, n_hidden)
            
            # Load state dict with strict=False to handle any minor differences
            model_dict = self.crnn_model.state_dict()
            pretrained_dict = {k: v for k, v in checkpoint['model_state_dict'].items() 
                             if k in model_dict and model_dict[k].shape == v.shape}
            
            if len(pretrained_dict) != len(checkpoint['model_state_dict']):
                logger.warning(f"Loaded {len(pretrained_dict)}/{len(checkpoint['model_state_dict'])} layers - some layer shapes may have changed")
            
            model_dict.update(pretrained_dict)
            self.crnn_model.load_state_dict(model_dict, strict=False)
            
            self.crnn_model.to(self.device)
            self.crnn_model.eval()
            
            logger.info(f"CRNN model loaded successfully! Character set size: {len(self.char_list)}")
            logger.info(f"Character set: {''.join(self.char_list[:20])}{'...' if len(self.char_list) > 20 else ''}")
            self.crnn_loaded = True
            
        except Exception as e:
            logger.error(f"Error loading CRNN model: {e}")
            self.crnn_loaded = False
            raise
    
    def detect_vehicles_and_plates(self, frame: np.ndarray) -> Tuple[List[Tuple[int, int, int, int]], List[Tuple[int, int, int, int]]]:
        """Detect vehicles and license plates in frame using YOLO"""
        if self.yolo_model is None:
            logger.debug("YOLO model is None, skipping detection")
            return [], []
            
        try:
            # Force CPU inference with torch.no_grad() for efficiency
            # Use reasonable confidence thresholds to avoid false positives
            with torch.no_grad():
                results = self.yolo_model(frame, conf=0.1, iou=Config.NMS_THRESHOLD, verbose=False, device='cpu')  # Lowered conf to 0.1 for debugging
            
            vehicles = []
            plates = []
            total_detections = 0
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    total_detections += len(boxes)
                    logger.info(f"🔍 YOLO found {len(boxes)} potential detections")  # Changed to INFO for visibility
                    
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                        conf = box.conf[0].cpu().numpy()
                        cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else -1
                        
                        # Filter by area and confidence
                        area = (x2 - x1) * (y2 - y1)
                        class_name = self.yolo_model.names[cls] if self.yolo_model and cls in self.yolo_model.names else f"Class {cls}"
                        logger.info(f"🔍 Detection: bbox=({x1},{y1},{x2},{y2}), conf={conf:.3f}, area={area}, class='{class_name}'")
                        
                        # Apply stricter thresholds for better accuracy
                        if cls in self.vehicle_class_indices:
                            if conf >= 0.15:
                                vehicles.append((x1, y1, x2, y2))
                                logger.info(f"✅ Accepted vehicle: conf={conf:.3f}, area={area}, class='{class_name}'")
                        elif cls == self.plate_class_idx:  # License plate
                            if conf >= 0.1 and area >= 200:
                                plates.append((x1, y1, x2, y2))
                                logger.info(f"✅ Accepted license plate: conf={conf:.3f}, area={area}")
                            else:
                                logger.info(f"❌ Rejected plate: conf={conf:.3f}, area={area} (min_conf=0.1, min_area=200)")
                        else:
                            logger.info(f"❓ Unknown class: {cls} ('{class_name}'), conf={conf:.3f}")
            
            # Filter plates to only keep those near vehicles (if enabled)
            if self.vehicle_plate_association_enabled:
                filtered_plates = self._filter_plates_near_vehicles(plates, vehicles, frame.shape)
                logger.info(f"🔍 Vehicle-plate association: {len(plates)} plates -> {len(filtered_plates)} filtered plates")
            else:
                filtered_plates = plates
                logger.info(f"🔍 Vehicle-plate association disabled: keeping all {len(plates)} plates")
            
            if total_detections == 0:
                logger.info("❌ No YOLO detections found in frame")
                # Save frame for debugging when no detections found
                if 'frame_id' in locals() and frame_id % 30 == 0:  # Save every 30th frame to avoid spam
                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                    debug_frame_path = Config.OUTPUT_DIR / f"debug_no_detections_frame_{frame_id}_{timestamp_str}.jpg"
                    cv2.imwrite(str(debug_frame_path), frame)
                    logger.warning(f"⚠️ YOLO DEBUG: Saved frame with no detections: {debug_frame_path}")
            else:
                logger.info(f"📊 YOLO SUMMARY: {total_detections} total detections, {len(vehicles)} cars, {len(plates)} plates -> {len(filtered_plates)} final plates")
            
            return vehicles, filtered_plates
            
        except Exception as e:
            logger.error(f"Error in plate detection: {e}")
            return [], []
    
    def _filter_plates_near_vehicles(self, plates: List[Tuple[int, int, int, int]], 
                                   vehicles: List[Tuple[int, int, int, int]], 
                                   frame_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int, int]]:
        """Filter license plates to only keep those that are spatially related to detected vehicles"""
        if not vehicles or not plates:
            return []
        
        filtered_plates = []
        frame_height, frame_width = frame_shape[:2]
        
        for plate_bbox in plates:
            px1, py1, px2, py2 = plate_bbox
            plate_center_x = (px1 + px2) // 2
            plate_center_y = (py1 + py2) // 2
            plate_area = (px2 - px1) * (py2 - py1)
            
            # Check if plate is associated with any vehicle
            is_associated = False
            min_distance = float('inf')
            
            for vehicle_bbox in vehicles:
                vx1, vy1, vx2, vy2 = vehicle_bbox
                vehicle_area = (vx2 - vx1) * (vy2 - vy1)
                
                # Check if plate is inside or near the vehicle bounding box
                # 1. Check if plate is inside vehicle bbox
                if (vx1 <= px1 <= vx2 and vy1 <= py1 <= vy2 and
                    vx1 <= px2 <= vx2 and vy1 <= py2 <= vy2):
                    is_associated = True
                    break
                
                # 2. Check if plate is reasonably close to vehicle
                vehicle_center_x = (vx1 + vx2) // 2
                vehicle_center_y = (vy1 + vy2) // 2
                
                distance = np.sqrt((plate_center_x - vehicle_center_x)**2 + 
                                 (plate_center_y - vehicle_center_y)**2)
                
                # Calculate dynamic threshold based on vehicle size
                vehicle_size = np.sqrt(vehicle_area)
                max_distance = min(300, vehicle_size * 1.5)  # Adaptive distance threshold
                
                if distance < max_distance:
                    min_distance = min(min_distance, distance)
                    # Additional checks for spatial relationship
                    
                    # 3. Check if plate size is reasonable relative to vehicle
                    size_ratio = plate_area / vehicle_area
                    if size_ratio < 0.5:  # Plate shouldn't be more than 50% of vehicle size
                        
                        # 4. Check if plate is in reasonable position relative to vehicle
                        # Plates are typically in the front or back of vehicles
                        relative_y = (plate_center_y - vy1) / (vy2 - vy1) if (vy2 - vy1) > 0 else 0.5
                        
                        # Accept if plate is in reasonable vertical position (not too high up)
                        if 0.2 <= relative_y <= 1.2:  # Allow some margin below vehicle for ground-level plates
                            is_associated = True
                            break
            
            if is_associated:
                filtered_plates.append(plate_bbox)
                logger.debug(f"Plate accepted: bbox=({px1},{py1},{px2},{py2}), min_distance={min_distance:.1f}")
            else:
                logger.debug(f"Plate rejected: bbox=({px1},{py1},{px2},{py2}), no nearby vehicle (min_distance={min_distance:.1f})")
        
        return filtered_plates
    
    def _preprocess_for_ocr(self, image: Image.Image) -> Image.Image:
        """Enhanced image preprocessing for license plates, matching training script exactly."""
        try:
            img_array = np.array(image)
            
            # Ensure it's RGB (matching training script)
            if len(img_array.shape) == 2:  # Grayscale image
                img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            elif len(img_array.shape) == 3 and img_array.shape[2] == 3:
                # Already RGB, but ensure proper format
                pass
            else:
                # Convert to RGB if needed
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            # Apply CLAHE to the L-channel of the LAB color space (exactly as in training)
            lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            img_array = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

            # Apply bilateral filter and sharpening (exactly as in training)
            img_array = cv2.bilateralFilter(img_array, 9, 75, 75)
            kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
            img_array = cv2.filter2D(img_array, -1, kernel)
            
            return Image.fromarray(img_array)
        except Exception as e:
            logger.warning(f"Failed to apply OCR preprocessing: {e}")
            return image  # Return original image on failure
            
    def preprocess_plate_image(self, image: np.ndarray) -> Optional[torch.Tensor]:
        """Preprocess plate image for CRNN model to exactly match training pipeline."""
        try:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3:
                pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            else:
                pil_image = Image.fromarray(image).convert('RGB')

            # Apply the exact same preprocessing as in training
            pil_image = self._preprocess_for_ocr(pil_image)

            # Use the exact same transforms as validation in training script
            transform = transforms.Compose([
                transforms.Resize((Config.OCR_IMG_HEIGHT, Config.OCR_IMG_WIDTH)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # RGB normalization
            ])
            
            tensor_image = transform(pil_image).unsqueeze(0)
            logger.debug(f"Preprocessed image shape: {tensor_image.shape}")
            return tensor_image

        except Exception as e:
            logger.error(f"Error preprocessing plate image: {e}")
            return None
    
    def decode_ctc_predictions(self, outputs) -> Tuple[List[str], List[float]]:
        """Decodes CTC predictions using the exact same logic as training script."""
        try:
            # Use the exact same decoding logic as in train_custom_crnn.py
            preds_idx = torch.argmax(outputs, dim=2)  # (seq_len, batch)
            preds_idx = preds_idx.transpose(0, 1).cpu().numpy()  # (batch, seq_len)
            
            decoded_texts = []
            confidences = []

            probs = torch.softmax(outputs, dim=2).transpose(0,1).cpu().detach().numpy() # (batch, seq_len, num_classes)

            for i in range(preds_idx.shape[0]): # Iterate over batch
                batch_preds = preds_idx[i]
                batch_probs = probs[i]
                
                text = []
                char_confidence = []
                last_char_idx = 0 
                for t in range(len(batch_preds)):
                    char_idx = batch_preds[t]
                    if char_idx != 0 and char_idx != last_char_idx: # Not blank and not repeated
                        if char_idx < len(self.char_list): # Check index bounds
                            text.append(self.char_list[char_idx])
                            char_confidence.append(batch_probs[t, char_idx])
                    last_char_idx = char_idx
                
                decoded_texts.append("".join(text))
                avg_conf = np.mean(char_confidence) if char_confidence else 0.0
                confidences.append(avg_conf)
                
            return decoded_texts, confidences
        except Exception as e:
            logger.error(f"Error in CTC decoding: {e}")
            return [], []
    
    def recognize_plate_text(self, plate_image: np.ndarray) -> Tuple[Optional[str], float]:
        """Recognize text from plate image using CRNN with exact training script logic"""
        if self.crnn_model is None or self.char_list is None:
            logger.warning("❌ CRNN model or character list not available")
            return None, 0.0
            
        try:
            logger.info(f"🔍 OCR: Processing plate image of shape: {plate_image.shape}")
            
            # Quality check: reject images that are too small or have poor quality
            if not self._is_good_plate_image(plate_image):
                logger.info("❌ OCR: Plate image quality check failed")
                return None, 0.0
            
            # Preprocess image using exact training pipeline
            tensor_image = self.preprocess_plate_image(plate_image)
            if tensor_image is None:
                logger.info("❌ OCR: Image preprocessing failed")
                return None, 0.0
            
            logger.info(f"🔍 OCR: Preprocessed tensor shape: {tensor_image.shape}")
            
            # Move to device
            tensor_image = tensor_image.to(self.device)
            
            # Run inference
            with torch.no_grad():
                outputs = self.crnn_model(tensor_image)  # (seq_len, batch, num_classes)
                logger.info(f"🔍 OCR: CRNN outputs shape: {outputs.shape}")
                logger.info(f"🔍 OCR: Outputs min/max: {outputs.min().item():.3f}/{outputs.max().item():.3f}")
                
                # Use the exact same decoding as training script
                decoded_texts, confidences = self.decode_ctc_predictions(outputs)
            
            logger.info(f"🔍 OCR: Decoded texts: {decoded_texts}, confidences: {confidences}")
            
            if decoded_texts and confidences:
                text = decoded_texts[0].upper().replace(" ", "")
                confidence = confidences[0]
                
                logger.info(f"🔍 OCR: Raw result: '{text}' (conf: {confidence:.3f})")
                
                # Validate text (basic license plate pattern)
                if self._validate_plate_text(text) and confidence >= Config.MIN_OCR_CONFIDENCE:
                    logger.info(f"✅ OCR: Result accepted: '{text}' (conf: {confidence:.3f})")
                    return text, confidence
                else:
                    logger.info(f"❌ OCR: Result rejected: '{text}' (conf: {confidence:.3f}, min_conf: {Config.MIN_OCR_CONFIDENCE})")
                    logger.info(f"❌ OCR: Validation passed: {self._validate_plate_text(text)}, Confidence check: {confidence >= Config.MIN_OCR_CONFIDENCE}")
            else:
                logger.info("❌ OCR: No decoded texts or confidences")
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"Error recognizing plate text: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return None, 0.0

    def _is_good_plate_image(self, image: np.ndarray) -> bool:
        """Check if the plate image has sufficient quality for OCR"""
        try:
            h, w = image.shape[:2]
            
            # Check minimum size - increased thresholds
            if h < 25 or w < 60:
                logger.debug(f"Image too small: {w}x{h}")
                return False
            
            # Check aspect ratio (plates are typically wider than they are tall)
            aspect_ratio = w / h
            if aspect_ratio < 2.0 or aspect_ratio > 6.0:  # Stricter aspect ratio
                logger.debug(f"Bad aspect ratio: {aspect_ratio:.2f}")
                return False
            
            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image
            
            # Check if image has sufficient contrast
            hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
            
            # Calculate contrast metric (standard deviation of pixel intensities)
            mean_intensity = np.mean(gray)
            contrast = np.std(gray)
            
            if contrast < 15:  # Very low contrast
                logger.debug(f"Low contrast: {contrast:.2f}")
                return False
            
            # Check if image is not too dark or too bright
            if mean_intensity < 30 or mean_intensity > 225:
                logger.debug(f"Poor brightness: {mean_intensity:.2f}")
                return False
            
            # Check for blur using Laplacian variance
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            if laplacian_var < 50:  # Very blurry
                logger.debug(f"Image too blurry: {laplacian_var:.2f}")
                return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error checking image quality: {e}")
            return False
    
    def _validate_plate_text(self, text: str) -> bool:
        """Enhanced validation for license plate text"""
        if not text or len(text) < 6:  # Minimum 6 characters for valid plate
            logger.debug(f"Plate validation failed: too short ({len(text) if text else 0} chars)")
            return False
        
        # Check if text contains reasonable characters for a license plate
        if not re.match(r'^[A-Z0-9]{6,12}$', text):
            logger.debug(f"Plate validation failed: invalid characters in '{text}'")
            return False
        
        # Reject obvious garbage patterns
        if self._is_garbage_text(text):
            logger.debug(f"Plate validation failed: garbage pattern '{text}'")
            return False
        
        # Check for reasonable license plate patterns
        # Indian plates: typically start with 2 letters, followed by 2 digits, then 1-2 letters, then 4 digits
        # Examples: KA31BR4210, TS15EX0371, GJ05SX1535
        indian_pattern = re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{1,3}[0-9]{4}$', text)
        if indian_pattern:
            logger.debug(f"Plate validation passed (Indian pattern): '{text}'")
            return True
        
        # International patterns - be more strict
        # Pattern 1: ABC1234 (3 letters + 4 numbers)
        if re.match(r'^[A-Z]{3}[0-9]{4}$', text):
            logger.debug(f"Plate validation passed (international pattern 1): '{text}'")
            return True
        
        # Pattern 2: AB12CD34 (2 letters + 2 numbers + 2 letters + 2 numbers)
        if re.match(r'^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{2}$', text):
            logger.debug(f"Plate validation passed (international pattern 2): '{text}'")
            return True
        
        # Pattern 3: 123ABC (3 numbers + 3 letters)
        if re.match(r'^[0-9]{3}[A-Z]{3}$', text):
            logger.debug(f"Plate validation passed (international pattern 3): '{text}'")
            return True
            
        logger.debug(f"Plate validation failed: doesn't match expected patterns '{text}'")
        return False

    def _is_garbage_text(self, text: str) -> bool:
        """Check if text appears to be garbage/random characters"""
        if not text:
            return True
        
        # Check for too many repeated characters
        for char in set(text):
            if text.count(char) > len(text) * 0.6:  # More than 60% same character
                return True
        
        # Check for common OCR garbage patterns
        garbage_patterns = [
            r'[FHD]{3,}',  # Too many F, H, D characters (common OCR errors)
            r'[089]{5,}',  # Too many similar-looking numbers
            r'^[ILOQ]{2,}', # Starting with confusing characters
            r'[XVW]{3,}',  # Too many wide characters
        ]
        
        for pattern in garbage_patterns:
            if re.search(pattern, text):
                return True
        
        # Check for reasonable distribution of letters vs numbers
        letters = sum(1 for c in text if c.isalpha())
        numbers = sum(1 for c in text if c.isdigit())
        
        # Reject if it's all letters or all numbers (except very specific cases)
        if letters == 0 or numbers == 0:
            # Allow some exceptions for valid patterns
            if not (re.match(r'^[A-Z]{6,8}$', text) or re.match(r'^[0-9]{6,8}$', text)):
                return False
        
        return False
    
    def detect_plate_type(self, plate_image: np.ndarray) -> str:
        """
        Detects the type/color of a license plate using HSV color masking and robust background analysis.
        """
        try:
            if plate_image is None or plate_image.size == 0:
                return "unknown"

            if len(plate_image.shape) != 3:
                return "white" # Assume white if not a color image

            hsv_image = cv2.cvtColor(plate_image, cv2.COLOR_BGR2HSV)
            height, width, _ = hsv_image.shape

            # --- Step 1: Robustly mask text regions ---
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            
            # Canny edge detection to find character boundaries
            edges = cv2.Canny(gray, 50, 200)
            
            # Dilate edges to connect broken parts of characters
            kernel = np.ones((3,3), np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours which are likely characters
            contours, _ = cv2.findContours(dilated_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            text_mask = np.zeros_like(gray)
            for cnt in contours:
                # Filter contours based on area and aspect ratio to isolate text
                x, y, w, h = cv2.boundingRect(cnt)
                aspect_ratio = h / w if w > 0 else 0
                if 1.0 < aspect_ratio < 6.0 and (height * 0.2) < h < (height * 0.9):
                    cv2.drawContours(text_mask, [cnt], -1, (255), -1)

            # Invert mask to get the background
            background_mask = cv2.bitwise_not(text_mask)
            background_area = cv2.countNonZero(background_mask)

            if background_area < (width * height * 0.2):
                logger.warning("Color detection unreliable: Not enough background detected.")
                # Fallback: use the whole image if masking fails
                background_mask = np.ones_like(gray) * 255
                background_area = width * height

            # --- Step 2: Define more generous HSV color ranges ---
            color_ranges = {
                "white": [(np.array([0, 0, 130]), np.array([180, 50, 255]))],
                "green": [(np.array([35, 30, 30]), np.array([95, 255, 255]))], # Expanded green to capture teals
                "yellow": [(np.array([15, 60, 60]), np.array([35, 255, 255]))],
                "red": [(np.array([0, 70, 50]), np.array([10, 255, 255])), (np.array([160, 70, 50]), np.array([180, 255, 255]))],
                "blue": [(np.array([96, 60, 50]), np.array([135, 255, 255]))], # Made blue stricter
            }

            # --- Step 3: Calculate color percentages on the background ---
            color_scores = {}
            for color, ranges in color_ranges.items():
                color_mask = cv2.inRange(hsv_image, ranges[0][0], ranges[0][1])
                if len(ranges) > 1:
                    mask2 = cv2.inRange(hsv_image, ranges[1][0], ranges[1][1])
                    color_mask = cv2.bitwise_or(color_mask, mask2)
                
                color_on_background = cv2.bitwise_and(color_mask, color_mask, mask=background_mask)
                score = (cv2.countNonZero(color_on_background) / background_area) * 100
                color_scores[color] = score
            
            logger.debug(f"🎨 Plate color scores: {color_scores}")

            # --- Step 4: Determine dominant color ---
            dominant_color = max(color_scores, key=color_scores.get)
            max_score = color_scores[dominant_color]
            
            # Additional check for gray plates that might be misclassified as blue
            if dominant_color == 'blue' and color_scores.get('white', 0) > 15:
                # Check average saturation of the plate
                avg_saturation = np.mean(hsv_image[:,:,1][background_mask > 0])
                if avg_saturation < 70:
                    logger.info("Reclassifying low-saturation 'blue' as white/gray.")
                    return "white"

            if max_score > 30.0: # Threshold of 30% of background area
                logger.info(f"✅ PLATE COLOR DETECTED: {dominant_color.upper()} (Score: {max_score:.2f}%)")
                return dominant_color
            else:
                logger.info(f"❓ PLATE COLOR UNKNOWN: Best guess '{dominant_color}' had low score {max_score:.2f}%.")
                return "unknown"

        except Exception as e:
            logger.error(f"Error in detect_plate_type: {e}")
            return "unknown"
    
    def get_plate_type_emoji(self, plate_type: str) -> str:
        """Get emoji representation for plate type"""
        type_emojis = {
            "white": "⚪",
            "green": "🟢", 
            "red": "🔴",
            "yellow": "🟡",
            "blue": "🔵",
            "unknown": "⚫"
        }
        return type_emojis.get(plate_type, "⚫")
    
    def is_unique_plate_detection(self, text: str, timestamp: float) -> bool:
        """Check if this is a unique plate detection (not a continuous sequence)"""
        try:
            # Check if we've seen this plate recently
            if text in self.last_detected_plates:
                time_since_last = timestamp - self.last_detected_plates[text]
                if time_since_last < self.unique_detection_window:
                    logger.debug(f"Plate '{text}' detected {time_since_last:.1f}s ago, skipping as continuous sequence")
                    return False
            
            # Update last detection time for this plate
            self.last_detected_plates[text] = timestamp
            logger.info(f"Unique detection: '{text}' (previous detection was >{self.unique_detection_window}s ago or never)")
            return True
            
        except Exception as e:
            logger.error(f"Error checking unique plate detection: {e}")
            return True  # Default to allowing detection
    
    def is_duplicate_detection(self, text: str, timestamp: float) -> bool:
        """Check if detection is a duplicate within time window"""
        current_time = timestamp
        
        # Remove old detections outside time window
        while self.recent_detections and (current_time - self.recent_detections[0].timestamp) > Config.DUPLICATE_TIME_WINDOW:
            self.recent_detections.popleft()
        
        # Check for similar text in recent detections
        for detection in self.recent_detections:
            if self._calculate_similarity(text, detection.text) >= Config.SIMILARITY_THRESHOLD:
                return True
        
        return False
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple character-based similarity
        if text1 == text2:
            return 1.0
        
        # Levenshtein distance based similarity
        max_len = max(len(text1), len(text2))
        if max_len == 0:
            return 1.0
        
        distance = self._levenshtein_distance(text1, text2)
        return 1.0 - (distance / max_len)
    
    def _levenshtein_distance(self, s1: str, s2: str) -> int:
        """Calculate Levenshtein distance between two strings"""
        if len(s1) < len(s2):
            return self._levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def process_frame(self, frame: np.ndarray, frame_id: int) -> List[Detection]:
        """Process a single frame and return detections"""
        detections = []
        timestamp = time.time()
        
        try:
            logger.debug(f"Processing frame {frame_id}, shape: {frame.shape}")
            
            # Frame skipping for real-time performance
            self.frame_skip_counter += 1
            #should_process_ocr = (self.frame_skip_counter % self.process_every_n_frames == 0)

            should_process_ocr = True
            
            # Always detect cars and plates for display, but limit OCR processing
            vehicle_boxes, plate_boxes = self.detect_vehicles_and_plates(frame)
            logger.debug(f"Frame {frame_id}: Found {len(vehicle_boxes)} cars, {len(plate_boxes)} plate boxes")
            
            # Log every 10th frame for debugging
            if frame_id % 10 == 0:
                logger.info(f"Frame {frame_id}: YOLO detection check - found {len(vehicle_boxes)} cars, {len(plate_boxes)} plates, OCR processing: {should_process_ocr}")
            
            # Update car tracking for parking vehicle detection
            if self.parking_detection_enabled:
                self.update_car_tracking(vehicle_boxes, timestamp)
                
                # Detect parked cars
                parked_car_detections = self.detect_parked_cars(timestamp)
                detections.extend(parked_car_detections)
                
                # Save parked car detections
                for parked_detection in parked_car_detections:
                    if should_process_ocr:  # Save only when processing
                        self._save_detection(frame, parked_detection)
            
            # Add regular car detections (for display only, save less frequently)
            for i, bbox in enumerate(vehicle_boxes):
                detection = Detection(
                    text="Car",
                    confidence=0.0,  # We don't track car confidence separately
                    bbox=bbox,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    detection_type="car"
                )
                detections.append(detection)
                
                # Save car detection image only occasionally to reduce I/O
                if should_process_ocr and i == 0:  # Save only first car per processing cycle
                    self._save_detection(frame, detection)
            
            # Process license plates for OCR (only when scheduled)
            if should_process_ocr and len(plate_boxes) > 0:
                logger.info(f"Frame {frame_id}: Processing {len(plate_boxes)} plates for OCR")
                
                for i, bbox in enumerate(plate_boxes):
                    x1, y1, x2, y2 = bbox
                    logger.debug(f"Processing plate {i+1}/{len(plate_boxes)}: bbox=({x1},{y1},{x2},{y2})")
                    
                    # Extract plate region
                    plate_region = frame[y1:y2, x1:x2]
                    if plate_region.size == 0:
                        logger.debug(f"Plate {i+1}: Empty region, skipping")
                        continue
                    
                    logger.debug(f"Plate {i+1}: Extracted region shape: {plate_region.shape}")
                    
                    # Recognize text
                    text, ocr_confidence = self.recognize_plate_text(plate_region)
                    logger.debug(f"Plate {i+1}: OCR result: '{text}' (conf: {ocr_confidence:.3f})")
                    
                    if text:
                        # Detect plate type/color
                        plate_type = self.detect_plate_type(plate_region)
                        logger.info(f"🎨 Plate {i+1}: Type detected: {plate_type.upper()}")
                        
                        # Save plate image for debugging (optional)
                        if text and plate_type != "unknown":
                            debug_plate_path = Config.OUTPUT_DIR / f"debug_plate_{text}_{plate_type}_{timestamp:.0f}.jpg"
                            cv2.imwrite(str(debug_plate_path), plate_region)
                            logger.info(f"💾 Saved debug plate image: {debug_plate_path.name}")
                        
                        # Check for unique detection (avoid continuous sequences)
                        if self.is_unique_plate_detection(text, timestamp) and not self.is_duplicate_detection(text, timestamp):
                            detection = Detection(
                                text=text,
                                confidence=ocr_confidence,
                                bbox=bbox,
                                timestamp=timestamp,
                                frame_id=frame_id,
                                detection_type="plate",
                                plate_type=plate_type
                            )
                            
                            detections.append(detection)
                            self.recent_detections.append(detection)
                            self.unique_plates.add(text)
                            self.total_detections += 1
                            
                            # Update plate type statistics
                            if plate_type in self.plate_type_counts:
                                self.plate_type_counts[plate_type] += 1
                            
                            # Save only car with label for high confidence (>= 0.91)
                            if ocr_confidence >= 0.91:
                                # Find the nearest car for this plate
                                nearest_car = None
                                min_distance = float('inf')
                                plate_center_x = (bbox[0] + bbox[2]) // 2
                                plate_center_y = (bbox[1] + bbox[3]) // 2
                                
                                for car_bbox in vehicle_boxes:
                                    car_center_x = (car_bbox[0] + car_bbox[2]) // 2
                                    car_center_y = (car_bbox[1] + car_bbox[3]) // 2
                                    distance = np.sqrt((plate_center_x - car_center_x)**2 + (plate_center_y - car_center_y)**2)
                                    if distance < min_distance:
                                        min_distance = distance
                                        nearest_car = car_bbox
                                
                                if nearest_car and min_distance <= 300:
                                    # Save only one image per car
                                    if text not in self.saved_car_plates:
                                        car_detection = Detection(
                                            text="Car",
                                            confidence=0.0,
                                            bbox=nearest_car,
                                            timestamp=timestamp,
                                            frame_id=frame_id,
                                            detection_type="car"
                                        )
                                        self._save_car_with_label(frame, car_detection, detection)
                                        self.saved_car_plates.add(text)
                                        logger.info(f"🎯 UNIQUE DETECTION: Saved car with label for {text} (conf: {ocr_confidence:.3f}) - FIRST TIME")
                                        self._log_detection_entry(detection)
                                    else:
                                        logger.info(f"🎯 UNIQUE DETECTION: {text} (conf: {ocr_confidence:.3f}) - Car image already saved")
                                else:
                                    # Fallback to old method if no car found - but still check if already saved
                                    if text not in self.saved_car_plates:
                                        self._save_detection(frame, detection)
                                        self.saved_car_plates.add(text)
                                        logger.info(f"🎯 UNIQUE: Saved plate only for {text} (conf: {ocr_confidence:.3f}) - No nearby car found")
                                        self._log_detection_entry(detection)
                                    else:
                                        logger.info(f"🎯 UNIQUE: {text} (conf: {ocr_confidence:.3f}) - Already saved")
                            else:
                                logger.info(f"🎯 UNIQUE: Plate detected but not saved: {text} (conf: {ocr_confidence:.3f}) - Below 0.91 threshold")
                            
                            logger.info(f"NEW UNIQUE Detection: {text} (conf: {ocr_confidence:.3f}) at frame {frame_id}")
                        else:
                            logger.debug(f"Plate {i+1}: Skipped '{text}' - continuous sequence or duplicate")
                    else:
                        logger.debug(f"Plate {i+1}: No text recognized")
            
            # Always show plate detection boxes even when not processing OCR
            elif len(plate_boxes) > 0:
                for bbox in plate_boxes:
                    detection = Detection(
                        text="Plate",
                        confidence=0.0,
                        bbox=bbox,
                        timestamp=timestamp,
                        frame_id=frame_id,
                        detection_type="plate_detected"
                    )
                    detections.append(detection)
        
        except Exception as e:
            logger.error(f"Error processing frame {frame_id}: {e}")
        
        logger.debug(f"Frame {frame_id} complete: {len(detections)} new detections")
        return detections
    
    def process_frame_with_zone(self, frame: np.ndarray, frame_id: int, detection_zone: List[Tuple[int, int]]) -> List[Detection]:
        """Process frame with stateful zone crossing based on precise line crossing."""
        try:
            vehicles, plates = self.detect_vehicles_and_plates(frame)
            timestamp = time.time()

            entry_line = detection_zone[:2]
            exit_line = detection_zone[2:]

            detections_in_zone = []
            
            logger.info(f"🚗 ZONE PROCESSING: Frame {frame_id}, found {len(vehicles)} vehicles, {len(plates)} plates")
            
            # ENHANCED DEBUGGING: Log detailed YOLO detection results
            if len(vehicles) > 0:
                logger.info(f"🚗 ZONE DEBUG: Vehicle bboxes: {vehicles}")
            if len(plates) > 0:
                logger.info(f"🏷️ ZONE DEBUG: Plate bboxes: {plates}")
            else:
                logger.warning(f"⚠️ ZONE DEBUG: No license plates detected by YOLO in this frame!")
            
            # Simple vehicle tracking for zone detection (using position similarity)
            current_frame_vehicles = []
            for i, bbox in enumerate(vehicles):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                current_frame_vehicles.append({
                    'id': f"v_{center_x}_{center_y}_{frame_id}",
                    'bbox': bbox,
                    'center': (center_x, center_y)
                })
            
            # Track vehicles across frames
            for vehicle in current_frame_vehicles:
                vehicle_id = vehicle['id']
                bbox = vehicle['bbox']
                x1, y1, x2, y2 = bbox
                
                # Use multiple points of vehicle for more reliable detection
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                bottom_center_x = center_x
                bottom_center_y = y2  # Bottom edge of vehicle
                front_center_x = center_x
                front_center_y = y1 + int((y2 - y1) * 0.7)  # 70% down from top
                
                # Check multiple points for more reliable detection
                entry_crossed = (
                    self._point_crossed_line((bottom_center_x, bottom_center_y), entry_line) or
                    self._point_crossed_line((center_x, center_y), entry_line) or
                    self._point_crossed_line((front_center_x, front_center_y), entry_line)
                )
                
                exit_crossed = (
                    self._point_crossed_line((bottom_center_x, bottom_center_y), exit_line) or
                    self._point_crossed_line((center_x, center_y), exit_line) or
                    self._point_crossed_line((front_center_x, front_center_y), exit_line)
                )
                
                logger.info(f"🚗 Vehicle {vehicle_id}: entry_crossed={entry_crossed}, exit_crossed={exit_crossed}")
                logger.debug(f"🚗 Vehicle points: center=({center_x},{center_y}), bottom=({bottom_center_x},{bottom_center_y}), front=({front_center_x},{front_center_y})")
                logger.debug(f"🚗 Entry line: {entry_line}, Exit line: {exit_line}")
                
                # Find similar vehicle from previous tracking
                similar_vehicle_id = self._find_similar_vehicle(vehicle['center'])
                
                # Process vehicle only if it has crossed entry line but not exit line
                if entry_crossed and not exit_crossed:
                    # Check if this vehicle is already being tracked in the zone
                    if similar_vehicle_id and similar_vehicle_id in self.zone_vehicle_states:
                        tracked_vehicle = self.zone_vehicle_states[similar_vehicle_id]
                        if tracked_vehicle.get('processed', False):
                            # Vehicle already processed, skip OCR
                            logger.info(f"🚗 Vehicle {similar_vehicle_id} already processed in zone.")
                            continue
                    
                    logger.info(f"🚗 Vehicle {vehicle_id} is IN the zone (crossed entry line but not exit line).")
                    logger.info(f"🚗 Vehicle bbox: ({x1},{y1},{x2},{y2}), center: ({center_x},{center_y})")
                    
                    in_zone_detection = Detection("Car (In-Zone)", 0.0, bbox, timestamp, frame_id, "car")
                    detections_in_zone.append(in_zone_detection)
                    
                    # OCR on associated plates within the vehicle's bounding box
                    logger.info(f"🔍 ZONE OCR: Checking {len(plates)} plates for vehicle {vehicle_id}")
                    
                    # ENHANCED DEBUGGING: Force OCR on all detected plates regardless of vehicle association
                    if len(plates) == 0:
                        logger.warning(f"⚠️ ZONE DEBUG: No plates detected by YOLO for vehicle in zone!")
                        logger.warning(f"⚠️ ZONE DEBUG: Vehicle bbox: {bbox}")
                        
                        # Save debug image of the vehicle region
                        debug_vehicle_region = frame[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                        timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                        debug_path = Config.OUTPUT_DIR / f"debug_no_plates_{timestamp_str}.jpg"
                        cv2.imwrite(str(debug_path), debug_vehicle_region)
                        logger.warning(f"⚠️ ZONE DEBUG: Saved vehicle region to {debug_path}")
                    
                    for i, plate_bbox in enumerate(plates):
                        logger.info(f"🔍 ZONE OCR: Plate {i+1}: bbox={plate_bbox}, vehicle bbox={bbox}")
                        
                        # Check if plate is within vehicle bounding box (with some tolerance)
                        px1, py1, px2, py2 = plate_bbox
                        vx1, vy1, vx2, vy2 = bbox
                        
                        # Expand vehicle bbox slightly for tolerance
                        tolerance = 100  # Increased tolerance for debugging
                        expanded_vx1 = vx1 - tolerance
                        expanded_vy1 = vy1 - tolerance
                        expanded_vx2 = vx2 + tolerance
                        expanded_vy2 = vy2 + tolerance
                        
                        plate_in_vehicle = (expanded_vx1 <= px1 and expanded_vy1 <= py1 and 
                                          expanded_vx2 >= px2 and expanded_vy2 >= py2)
                        
                        logger.info(f"🔍 ZONE OCR: Plate {i+1} in vehicle: {plate_in_vehicle} (tolerance: {tolerance})")
                        logger.info(f"🔍 ZONE OCR: Expanded vehicle bbox: ({expanded_vx1},{expanded_vy1},{expanded_vx2},{expanded_vy2})")
                        
                        # FORCE PROCESSING ALL PLATES FOR DEBUGGING
                        should_process_plate = plate_in_vehicle or True  # Force process all plates
                        
                        if should_process_plate:
                            if not plate_in_vehicle:
                                logger.warning(f"⚠️ ZONE DEBUG: Processing plate {i+1} even though it's outside vehicle bbox (for debugging)")
                            
                            plate_region = frame[plate_bbox[1]:plate_bbox[3], plate_bbox[0]:plate_bbox[2]]
                            logger.info(f"🔍 ZONE OCR: Extracted plate region shape: {plate_region.shape}")
                            
                            # Save debug image of plate region
                            if plate_region.size > 0:
                                timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                                debug_plate_path = Config.OUTPUT_DIR / f"debug_zone_plate_{i+1}_{timestamp_str}.jpg"
                                cv2.imwrite(str(debug_plate_path), plate_region)
                                logger.info(f"💾 ZONE DEBUG: Saved plate region to {debug_plate_path}")
                            
                            text, conf = self.recognize_plate_text(plate_region)
                            logger.info(f"🔍 ZONE OCR: Result: '{text}' (conf: {conf:.3f})")
                            
                            # ENHANCED DEBUGGING: Lower confidence threshold and detailed logging
                            logger.info(f"🔍 ZONE DEBUG: OCR text='{text}', conf={conf:.3f}, min_conf={Config.MIN_OCR_CONFIDENCE}")
                            logger.info(f"🔍 ZONE DEBUG: Already processed: {text in self.zone_processed_plates if text else 'N/A'}")
                            
                            # Temporarily lower confidence threshold for debugging
                            debug_min_conf = 0.1  # Much lower threshold for debugging
                            
                            if text and conf >= debug_min_conf and text not in self.zone_processed_plates and self.is_unique_plate_detection(text, timestamp):
                                # Detect plate type/color
                                plate_type = self.detect_plate_type(plate_region)
                                logger.info(f"🔍 ZONE: Plate type detected: {plate_type}")
                                
                                # Mark this plate as processed in the current zone session
                                self.zone_processed_plates.add(text)
                                
                                plate_det = Detection(text, conf, plate_bbox, timestamp, frame_id, "plate", plate_type)
                                detections_in_zone.append(plate_det)
                                self.recent_detections.append(plate_det)
                                self.unique_plates.add(text)
                                self.total_detections += 1
                                
                                # Update plate type statistics
                                if plate_type in self.plate_type_counts:
                                    self.plate_type_counts[plate_type] += 1
                                
                                # Mark vehicle as processed
                                tracking_id = similar_vehicle_id or vehicle_id
                                self.zone_vehicle_states[tracking_id] = {
                                    'state': 'in_zone',
                                    'last_seen': timestamp,
                                    'processed': True,
                                    'plate_text': text,
                                    'entry_time': timestamp
                                }
                                
                                logger.info(f"✅ ZONE SUCCESS: Detected plate '{text}' with confidence {conf:.3f}")
                                
                                if conf >= 0.91 and text not in self.saved_car_plates:
                                    self._save_car_with_label(frame, in_zone_detection, plate_det)
                                    self.saved_car_plates.add(text)
                                    logger.info(f"🎯 SAVED: Car with label for {text}")
                                    self._log_detection_entry(plate_det)
                            elif text in self.zone_processed_plates:
                                logger.info(f"🔍 ZONE OCR: Plate {text} already processed in current zone session")
                            elif not text:
                                logger.warning(f"⚠️ ZONE OCR: No text recognized from plate region (shape: {plate_region.shape})")
                                # Save the plate region that failed OCR for analysis
                                if plate_region.size > 0:
                                    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
                                    failed_ocr_path = Config.OUTPUT_DIR / f"debug_failed_ocr_{timestamp_str}.jpg"
                                    cv2.imwrite(str(failed_ocr_path), plate_region)
                                    logger.warning(f"⚠️ ZONE OCR: Saved failed OCR region to {failed_ocr_path}")
                            elif conf < debug_min_conf:
                                logger.warning(f"⚠️ ZONE OCR: Plate '{text}' confidence too low: {conf:.3f} < {debug_min_conf}")
                            else:
                                logger.info(f"🔍 ZONE OCR: Plate '{text}' failed uniqueness check")
                                
                elif exit_crossed:
                    logger.info(f"🚗 Vehicle {vehicle_id} crossed exit line - stopping detection.")
                    similar_vehicle_id = self._find_similar_vehicle(vehicle['center'])
                    if similar_vehicle_id and similar_vehicle_id in self.zone_vehicle_states:
                        vehicle_data = self.zone_vehicle_states[similar_vehicle_id]
                        if vehicle_data.get('processed'):
                            plate_text = vehicle_data.get('plate_text', 'UNKNOWN')
                            entry_time = vehicle_data.get('entry_time', timestamp)
                            duration = timestamp - entry_time
                            
                            exit_detection = Detection(
                                text=plate_text,
                                confidence=1.0,
                                bbox=bbox,
                                timestamp=timestamp,
                                frame_id=frame_id,
                                detection_type="exit",
                                plate_type=f"duration: {duration:.1f}s"
                            )
                            detections_in_zone.append(exit_detection)
                            self._log_detection_entry(exit_detection)
                        
                        del self.zone_vehicle_states[similar_vehicle_id]
                    
                    if hasattr(self, 'zone_processed_plates'):
                        remaining_vehicles = [v for v in self.zone_vehicle_states.values() if v.get('state') == 'in_zone']
                        if len(remaining_vehicles) == 0:
                            self.zone_processed_plates.clear()
                            logger.info("🔍 Cleared zone processed plates - no vehicles remaining in zone")

            # Clean up old vehicle states
            current_time = timestamp
            vehicles_to_remove = []
            for vid, vstate in self.zone_vehicle_states.items():
                if current_time - vstate['last_seen'] > 10.0:  # 10 seconds timeout
                    vehicles_to_remove.append(vid)
            
            for vid in vehicles_to_remove:
                del self.zone_vehicle_states[vid]

            logger.info(f"🚗 ZONE RESULT: Frame {frame_id} returning {len(detections_in_zone)} detections")
            return detections_in_zone

        except Exception as e:
            logger.error(f"Error processing frame with zone: {e}", exc_info=True)
            return self.process_frame(frame, frame_id)
    
    def _line_crossed(self, p1: Tuple[int, int], p2: Tuple[int, int], line_p1: Tuple[int, int], line_p2: Tuple[int, int]) -> bool:
        """Checks if the line segment from p1 to p2 intersects the line segment from line_p1 to line_p2."""
        try:
            # Using vector cross products to check for intersection
            def cross_product(v1, v2):
                return v1[0] * v2[1] - v1[1] * v2[0]

            def subtract(v1, v2):
                return (v1[0] - v2[0], v1[1] - v2[1])

            r = subtract(p2, p1)
            s = subtract(line_p2, line_p1)
            
            r_cross_s = cross_product(r, s)
            
            # If lines are collinear or parallel, r_cross_s will be close to 0
            if abs(r_cross_s) < 1e-10:
                return False

            q_minus_p = subtract(line_p1, p1)
            t = cross_product(q_minus_p, s) / r_cross_s
            u = cross_product(q_minus_p, r) / r_cross_s

            # Intersection if 0 <= t <= 1 and 0 <= u <= 1
            return 0.0 <= t <= 1.0 and 0.0 <= u <= 1.0
        except Exception as e:
            logger.error(f"Error in _line_crossed: {e}")
            return False

    def _box_intersects_line(self, bbox: Tuple[int, int, int, int], line: List[Tuple[int, int]]) -> bool:
        """Checks if a bounding box intersects a line segment using cv2.clipLine."""
        try:
            x1, y1, x2, y2 = bbox
            # cv2.clipLine requires rect in (x, y, w, h) format
            rect = (x1, y1, x2 - x1, y2 - y1)
            p1, p2 = line
            
            # The function returns a boolean indicating if the line segment is visible at all
            intersects, _, _ = cv2.clipLine(rect, p1, p2)
            return intersects
        except Exception as e:
            logger.error(f"Error in _box_intersects_line: {e}")
            return False

    def _point_crossed_line(self, point: Tuple[int, int], line: List[Tuple[int, int]]) -> bool:
        """Check if a point has crossed a line with improved logic."""
        try:
            px, py = point
            (x1, y1), (x2, y2) = line
            
            # Calculate the y-coordinate on the line at the point's x-coordinate
            if abs(x2 - x1) < 1:  # Nearly vertical line
                # For vertical lines, check if point is within x range and below the line
                if abs(px - x1) <= 10:  # Within 10 pixels of the line
                    line_y = min(y1, y2)
                    crossed = py >= line_y
                else:
                    crossed = False
            else:
                # Linear interpolation to find y at px
                slope = (y2 - y1) / (x2 - x1)
                line_y = y1 + slope * (px - x1)
                
                # Point has "crossed" the line if it's below the line
                # Add tolerance and ensure point is within reasonable x range of the line
                tolerance = 15  # Increased tolerance
                x_min, x_max = min(x1, x2), max(x1, x2)
                
                # Check if point is within the x range of the line (with some extension)
                x_margin = abs(x2 - x1) * 0.1  # 10% margin on each side
                if x_min - x_margin <= px <= x_max + x_margin:
                    crossed = py >= (line_y - tolerance)
                else:
                    crossed = False
            
            logger.debug(f"Point ({px},{py}) vs Line ({x1},{y1})-({x2},{y2}): crossed={crossed}")
            return crossed
            
        except Exception as e:
            logger.error(f"Error in _point_crossed_line: {e}")
            return False

    def _find_similar_vehicle(self, current_center: Tuple[int, int]) -> Optional[str]:
        """Find a similar vehicle from previous tracking based on position."""
        try:
            current_x, current_y = current_center
            min_distance = float('inf')
            similar_vehicle_id = None
            
            for vehicle_id, vehicle_data in self.zone_vehicle_states.items():
                if 'last_seen' in vehicle_data:
                    # Extract position from vehicle_id (simplified approach)
                    try:
                        # Parse vehicle_id format: "v_{center_x}_{center_y}_{frame_id}"
                        parts = vehicle_id.split('_')
                        if len(parts) >= 3:
                            prev_x = int(parts[1])
                            prev_y = int(parts[2])
                            
                            # Calculate distance
                            distance = np.sqrt((current_x - prev_x)**2 + (current_y - prev_y)**2)
                            
                            # Consider it the same vehicle if within 100 pixels
                            if distance < 100 and distance < min_distance:
                                min_distance = distance
                                similar_vehicle_id = vehicle_id
                    except (ValueError, IndexError):
                        continue
            
            return similar_vehicle_id
            
        except Exception as e:
            logger.error(f"Error finding similar vehicle: {e}")
            return None

    def _point_to_line_distance(self, px: int, py: int, x1: int, y1: int, x2: int, y2: int) -> float:
        """Calculate the shortest distance from a point to a line segment"""
        try:
            # Line length squared
            line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
            
            if line_length_sq == 0:
                # Line is actually a point
                return np.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            
            # Parameter t that represents the closest point on the line
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
            
            # Closest point on the line
            closest_x = x1 + t * (x2 - x1)
            closest_y = y1 + t * (y2 - y1)
            
            # Distance from point to closest point on line
            return np.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
            
        except Exception as e:
            logger.error(f"Error calculating point to line distance: {e}")
            return float('inf')
    
    def update_car_tracking(self, vehicle_boxes: List[Tuple[int, int, int, int]], timestamp: float):
        """Update tracking information for still car detection"""
        try:
            current_cars = {}
            
            # Match current detections with existing tracked cars
            for i, bbox in enumerate(vehicle_boxes):
                x1, y1, x2, y2 = bbox
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                area = (x2 - x1) * (y2 - y1)
                
                # Find closest existing tracked car
                closest_car_id = None
                min_distance = float('inf')
                
                for car_id, car_data in self.car_tracking.items():
                    last_center = car_data['centers'][-1]
                    distance = np.sqrt((center_x - last_center[0])**2 + (center_y - last_center[1])**2)
                    
                    # Check if car is close enough and similar size
                    area_ratio = min(area, car_data['area']) / max(area, car_data['area'])
                    if distance < 50 and area_ratio > 0.7:  # Within 50 pixels and similar size
                        if distance < min_distance:
                            min_distance = distance
                            closest_car_id = car_id
                
                if closest_car_id is not None:
                    # Update existing car
                    car_data = self.car_tracking[closest_car_id]
                    car_data['centers'].append((center_x, center_y))
                    car_data['bboxes'].append(bbox)
                    car_data['timestamps'].append(timestamp)
                    car_data['area'] = area
                    car_data['last_seen'] = timestamp
                    
                    # Keep only recent positions (last 30 frames)
                    if len(car_data['centers']) > 30:
                        car_data['centers'] = car_data['centers'][-30:]
                        car_data['bboxes'] = car_data['bboxes'][-30:]
                        car_data['timestamps'] = car_data['timestamps'][-30:]
                    
                    current_cars[closest_car_id] = True
                else:
                    # Create new tracked car
                    new_car_id = f"car_{len(self.car_tracking)}_{timestamp:.0f}"
                    self.car_tracking[new_car_id] = {
                        'centers': [(center_x, center_y)],
                        'bboxes': [bbox],
                        'timestamps': [timestamp],
                        'area': area,
                        'first_seen': timestamp,
                        'last_seen': timestamp
                    }
                    current_cars[new_car_id] = True
            
            # Remove cars that haven't been seen for too long
            cars_to_remove = []
            for car_id, car_data in self.car_tracking.items():
                if timestamp - car_data['last_seen'] > 5.0:  # 5 seconds
                    cars_to_remove.append(car_id)
            
            for car_id in cars_to_remove:
                del self.car_tracking[car_id]
                if car_id in self.parked_cars:
                    del self.parked_cars[car_id]
            
        except Exception as e:
            logger.error(f"Error updating car tracking: {e}")
    
    def detect_parked_cars(self, timestamp: float) -> List[Detection]:
        """Detect cars that have been parked for the threshold time"""
        parked_detections = []
        
        try:
            for car_id, car_data in self.car_tracking.items():
                # Check if car has been tracked long enough
                time_tracked = timestamp - car_data['first_seen']
                if time_tracked < self.parking_time_threshold:
                    continue
                
                # Check if car has been relatively stationary
                centers = car_data['centers']
                if len(centers) < 5:  # Need at least 5 positions to check
                    continue
                
                # Calculate movement variance
                recent_centers = centers[-10:]  # Check last 10 positions
                x_coords = [c[0] for c in recent_centers]
                y_coords = [c[1] for c in recent_centers]
                
                x_variance = np.var(x_coords) if len(x_coords) > 1 else 0
                y_variance = np.var(y_coords) if len(y_coords) > 1 else 0
                movement_variance = x_variance + y_variance
                
                # Car is considered parked if movement variance is low
                if movement_variance < 100:  # Threshold for parking
                    if car_id not in self.parked_cars:
                        # Mark as parked car
                        self.parked_cars[car_id] = {
                            'detected_at': timestamp,
                            'bbox': car_data['bboxes'][-1],
                            'duration': time_tracked
                        }
                        
                        # Create detection for parked car
                        bbox = car_data['bboxes'][-1]
                        detection = Detection(
                            text=f"Parked Vehicle ({time_tracked:.1f}s)",
                            confidence=0.95,  # High confidence for parked cars
                            bbox=bbox,
                            timestamp=timestamp,
                            frame_id=int(timestamp * 30),  # Approximate frame
                            detection_type="parked_car"
                        )
                        parked_detections.append(detection)
                        
                        logger.info(f"Parked vehicle detected: {car_id} stationary for {time_tracked:.1f}s")
                    else:
                        # Update duration for existing parked car
                        self.parked_cars[car_id]['duration'] = time_tracked
                        
                        # Update detection text with current duration
                        bbox = car_data['bboxes'][-1]
                        detection = Detection(
                            text=f"Parked Vehicle ({time_tracked:.1f}s)",
                            confidence=0.95,
                            bbox=bbox,
                            timestamp=timestamp,
                            frame_id=int(timestamp * 30),
                            detection_type="parked_car"
                        )
                        parked_detections.append(detection)
                else:
                    # Car is moving again, remove from parked cars
                    if car_id in self.parked_cars:
                        del self.parked_cars[car_id]
                        logger.info(f"Car {car_id} is moving again")
        
        except Exception as e:
            logger.error(f"Error detecting parked cars: {e}")
        
        return parked_detections
    
    def _save_essential_detection(self, frame: np.ndarray, detection: Detection):
        """Save only essential detection images - plate crop and vehicle context"""
        try:
            if not Config.SAVE_DETECTIONS:
                return
            x1, y1, x2, y2 = detection.bbox
            detected_region = frame[y1:y2, x1:x2]
            
            timestamp_str = datetime.fromtimestamp(detection.timestamp).strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            if detection.detection_type == "plate":
                # 1. Save cropped license plate
                plate_text = detection.text.replace("🎯", "")  # Remove target emoji
                plate_filename = f"plate_{plate_text}_{timestamp_str}_conf{detection.confidence:.3f}.jpg"
                plate_filepath = Config.OUTPUT_DIR / plate_filename
                cv2.imwrite(str(plate_filepath), detected_region)
                logger.debug(f"Saved plate crop: {plate_filename}")
                
                # 2. Save vehicle context around license plate
                self._save_vehicle_context(frame, detection, timestamp_str, plate_text)
                
                logger.debug(f"Saved essential detection for: {plate_text}")
            
        except Exception as e:
            logger.error(f"Error saving essential detection: {e}")
    
    def _save_detection(self, frame: np.ndarray, detection: Detection):
        """Save detection image to output directory with improved labeling"""
        # This function is now OBSOLETE. All saving is handled by _save_car_with_label.
        # Kept for potential future debugging but should not be called in production.
        pass
    
    def _save_labeled_image(self, frame: np.ndarray, detection: Detection, timestamp_str: str):
        """Save full frame with detection bounding box and label"""
        # This function is now OBSOLETE.
        pass
    
    def _save_vehicle_context(self, frame: np.ndarray, plate_detection: Detection, timestamp_str: str, plate_text: str):
        """Save vehicle context around license plate detection"""
        # This function is now OBSOLETE.
        pass
    
    def _save_car_with_label(self, frame: np.ndarray, car_detection: Detection, plate_detection: Detection):
        """Save only the car image with license plate label overlay"""
        try:
            if not Config.SAVE_DETECTIONS:
                return
            # Get car bounding box
            car_x1, car_y1, car_x2, car_y2 = car_detection.bbox
            car_region = frame[car_y1:car_y2, car_x1:car_x2].copy()
            
            # Get plate bounding box relative to car region
            plate_x1, plate_y1, plate_x2, plate_y2 = plate_detection.bbox
            rel_plate_x1 = max(0, plate_x1 - car_x1)
            rel_plate_y1 = max(0, plate_y1 - car_y1)
            rel_plate_x2 = min(car_x2 - car_x1, plate_x2 - car_x1)
            rel_plate_y2 = min(car_y2 - car_y1, plate_y2 - car_y1)
            
            # Draw plate bounding box on car image
            cv2.rectangle(car_region, (rel_plate_x1, rel_plate_y1), (rel_plate_x2, rel_plate_y2), (0, 255, 0), 2)
            
            # Add plate text label
            plate_text = plate_detection.text.replace("🎯", "")  # Remove emoji
            label = f"{plate_text} ({plate_detection.confidence:.2f})"
            label_y = max(20, rel_plate_y1 - 10)
            
            cv2.putText(car_region, label, (rel_plate_x1, label_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            # Save the labeled car image
            plate_text_for_filename = plate_detection.text.replace("🎯", "")
            timestamp_str = datetime.fromtimestamp(plate_detection.timestamp).strftime('%Y%m%d_%H%M%S_%f')[:-3]
            filename = f"car_{plate_text_for_filename}_{timestamp_str}_conf{plate_detection.confidence:.3f}.jpg"
            filepath = Config.OUTPUT_DIR / filename
            cv2.imwrite(str(filepath), car_region)
            
            logger.debug(f"Saved car with label: {filename}")
            
        except Exception as e:
            logger.error(f"Error saving car with label: {e}")

    def _log_detection_entry(self, detection: Detection):
        """Logs a confirmed unique detection to a file."""
        try:
            log_file = Config.OUTPUT_DIR / "detection_log.jsonl"
            log_entry = {
                "timestamp": datetime.fromtimestamp(detection.timestamp).isoformat(),
                "event_type": detection.detection_type,
                "plate_number": detection.text,
                "confidence": round(detection.confidence, 4),
                "plate_type": detection.plate_type,
                "frame_id": detection.frame_id
            }
            with open(log_file, 'a') as f:
                import json
                f.write(json.dumps(log_entry) + '\n')
            logger.info(f"💾 Logged new entry: {detection.text}")
        except Exception as e:
            logger.error(f"Failed to log detection entry: {e}")

class VideoThread(QThread):
    """Thread for video processing"""
    
    frame_ready = pyqtSignal(np.ndarray, list)  # frame, detections
    stats_updated = pyqtSignal(dict)  # stats dictionary
    models_loaded = pyqtSignal(bool, bool)  # yolo_loaded, crnn_loaded
    
    def __init__(self, source_type='camera', source_path=None, start_frame=0, detection_point=None, 
                 parking_enabled=False, parking_time_threshold=3.0, camera_index=0, camera_config=None):
        super().__init__()
        self.source_type = source_type
        self.source_path = source_path
        self.camera_index = camera_index
        self.camera_config = camera_config or {}
        self.running = False
        self.paused = False
        self.playback_speed = 1.0  # 1x speed by default
        self.start_frame = start_frame
        self.detection_point = detection_point
        self.cap = None  # Video capture object
        
        # Thread-safe seeking
        self.seek_request_seconds = None
        self.seek_lock = threading.Lock()
        
        self.anpr_processor = ANPRProcessor()
        self.anpr_processor.parking_detection_enabled = parking_enabled
        self.anpr_processor.parking_time_threshold = parking_time_threshold
        self.frame_count = 0
        
    def run(self):
        """Main video processing loop"""
        try:
            # Load models
            if not self.anpr_processor.load_models():
                logger.error("Failed to load models")
                return
            
            # Emit model status
            self.models_loaded.emit(self.anpr_processor.yolo_loaded, self.anpr_processor.crnn_loaded)
            
            # Initialize video source
            if self.source_type == 'camera':
                cap = configure_camera(
                    self.camera_index,
                    self.camera_config.get('width', Config.CAMERA_WIDTH),
                    self.camera_config.get('height', Config.CAMERA_HEIGHT),
                    self.camera_config.get('fps', Config.CAMERA_FPS)
                )
                if cap is None:
                    logger.error(f"Failed to configure camera {self.camera_index}")
                    cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)  # Fallback to basic camera
            else:
                cap = cv2.VideoCapture(self.source_path)
            
            self.cap = cap  # Store reference for seeking
            
            if not cap.isOpened():
                logger.error(f"Failed to open video source: {self.source_path}")
                return
            
            # Set starting frame if specified
            if self.start_frame > 0 and self.source_type == 'video':
                cap.set(cv2.CAP_PROP_POS_FRAMES, self.start_frame)
                logger.info(f"Starting from frame {self.start_frame}")
            
            self.running = True
            fps = cap.get(cv2.CAP_PROP_FPS) or 30
            
            logger.info(f"Started video processing (FPS: {fps}, Speed: {self.playback_speed}x)")
            if self.detection_point:
                logger.info(f"Detection zone active: {self.detection_point[0]} to {self.detection_point[3]}")
            if self.anpr_processor.parking_detection_enabled:
                logger.info(f"Parking vehicle detection enabled (threshold: {self.anpr_processor.parking_time_threshold}s)")
            
            start_time = time.time()
            
            while self.running:
                if self.paused:
                    self.msleep(50)  # Sleep when paused to avoid high CPU usage
                    continue

                # Thread-safe seek handling
                with self.seek_lock:
                    if self.seek_request_seconds is not None:
                        if self.cap and self.source_type == 'video':
                            fps = self.cap.get(cv2.CAP_PROP_FPS) or 30
                            total_frames = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
                            current_frame = self.cap.get(cv2.CAP_PROP_POS_FRAMES)
                            
                            frames_to_seek = int(self.seek_request_seconds * fps)
                            new_frame_pos = current_frame + frames_to_seek
                            
                            # Clamp to valid frame range
                            new_frame_pos = max(0, min(new_frame_pos, total_frames - 1))
                            
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, new_frame_pos)
                            self.frame_count = int(new_frame_pos) # Update frame count
                            logger.info(f"Seeked to frame {self.frame_count}")
                        
                        # Reset request
                        self.seek_request_seconds = None

                loop_start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    if self.source_type == 'video':
                        # Restart video and reset trackers
                        logger.info("Video ended, restarting from beginning.")
                        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        self.frame_count = 0
                        self.anpr_processor.zone_vehicle_states.clear()
                        self.anpr_processor.saved_car_plates.clear()
                        continue
                    else:
                        break
                
                # Process frame
                if self.detection_point:
                    detections = self.anpr_processor.process_frame_with_zone(frame, self.frame_count, self.detection_point)
                else:
                    detections = self.anpr_processor.process_frame(frame, self.frame_count)
                
                self.frame_ready.emit(frame, detections)
                
                # Update stats
                stats = {
                    'frame_count': self.frame_count,
                    'total_detections': self.anpr_processor.total_detections,
                    'unique_plates': len(self.anpr_processor.unique_plates),
                    'recent_detections': len(self.anpr_processor.recent_detections),
                    'tracked_cars': len(self.anpr_processor.car_tracking) if hasattr(self.anpr_processor, 'car_tracking') else 0,
                    'parked_cars': len(self.anpr_processor.parked_cars) if hasattr(self.anpr_processor, 'parked_cars') else 0,
                    'white_plates': self.anpr_processor.plate_type_counts.get('white', 0),
                    'green_plates': self.anpr_processor.plate_type_counts.get('green', 0),
                    'red_plates': self.anpr_processor.plate_type_counts.get('red', 0)
                }
                self.stats_updated.emit(stats)
                
                self.frame_count += 1
                
                # Improved frame rate control logic with better speed handling
                processing_time = time.time() - loop_start_time
                
                # Handle different playback speeds
                if self.playback_speed > 0:
                    target_duration = 1.0 / (fps * self.playback_speed)
                    sleep_duration = target_duration - processing_time
                    
                    if sleep_duration > 0:
                        self.msleep(max(1, int(sleep_duration * 1000)))
                    else:
                        # For very high speeds, yield occasionally to prevent UI freezing
                        if self.frame_count % 10 == 0:
                            self.msleep(1)
                else:
                    # Fallback if speed is invalid
                    self.msleep(33)  # ~30 FPS default
            
            cap.release()
            self.cap = None
            logger.info("Video processing stopped")
            
        except Exception as e:
            logger.error(f"Error in video thread: {e}")
    
    def stop(self):
        """Stop video processing"""
        self.running = False
        self.wait()
    
    def pause(self):
        """Pause video processing"""
        self.paused = True
    
    def resume(self):
        """Resume video processing"""
        self.paused = False

    def request_seek(self, seconds: int):
        """Request to seek the video by a number of seconds."""
        with self.seek_lock:
            self.seek_request_seconds = seconds
        logger.info(f"Seek requested: {seconds} seconds")

class MainWindow(QMainWindow):
    """Main GUI window"""
    
    def __init__(self):
        super().__init__()
        self.video_thread = None
        self.current_detections = []
        self.all_detections_history = []  # Store all detections for live feed
        self.all_events_history = [] # Store entry/exit events for log
        
        # Timer for live feed updates
        self.live_feed_timer = QTimer()
        self.live_feed_timer.timeout.connect(self.update_live_feed)
        self.live_feed_timer.timeout.connect(self.update_events_feed)
        self.live_feed_timer.setInterval(500)  # Update every 500ms
        
        # Point detection variables
        self.detection_points = []  # Will store [point_a, point_b]
        self.selecting_points = False
        self.points_complete = False
        self.current_frame_for_point = None
        self.first_frame_loaded = False
        
        self.setup_ui()
        self.setup_connections()
        
        # Connect mouse events for point selection
        self.video_label.mousePressEvent = self.mouse_press_event
        
        # Initialize point display
        self.update_point_display()
        
        # Detect available cameras on startup
        self.available_cameras = []
        self.detect_available_cameras()
        
    def setup_ui(self):
        """Setup user interface"""
        self.setWindowTitle("ANPR Live Detection System")
        self.setGeometry(100, 100, 1400, 900)
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # Main layout - use a wrapper layout to hold the splitter
        wrapper_layout = QHBoxLayout(central_widget)
        wrapper_layout.setSpacing(0)
        wrapper_layout.setContentsMargins(10, 10, 10, 10)
        
        # Splitter for draggable resizing between video and right panel
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(6)
        self.main_splitter.setStyleSheet("""
            QSplitter::handle {
                background-color: #555;
                border: 1px solid #888;
                border-radius: 2px;
                margin: 2px 0px;
            }
            QSplitter::handle:hover {
                background-color: #4CAF50;
            }
            QSplitter::handle:pressed {
                background-color: #388E3C;
            }
        """)
        wrapper_layout.addWidget(self.main_splitter)
        
        # Left panel - Video display
        left_panel = QVBoxLayout()
        
        # Video display
        self.video_label = QLabel("No video source selected")
        self.video_label.setMinimumSize(640, 480)  # Reduced minimum size
        self.video_label.setStyleSheet("border: 2px solid gray; background-color: black;")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setScaledContents(False)  # Ensure proper aspect ratio
        # Set size policy to expand
        from PyQt5.QtWidgets import QSizePolicy
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        left_panel.addWidget(self.video_label)
        
        # Instructions label for polygon drawing
        self.instructions_label = QLabel("")
        self.instructions_label.setAlignment(Qt.AlignCenter)
        self.instructions_label.setStyleSheet("color: yellow; font-weight: bold; background-color: rgba(0,0,0,100);")
        self.instructions_label.hide()
        left_panel.addWidget(self.instructions_label)
        
        # Video controls
        controls_layout = QHBoxLayout()
        controls_layout.setSpacing(5)
        
        self.source_combo = QComboBox()
        self.source_combo.addItem("Camera")
        self.source_combo.addItem("Video File")
        self.source_combo.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
        controls_layout.addWidget(QLabel("Source:"))
        controls_layout.addWidget(self.source_combo)
        
        # Camera selection
        self.camera_combo = QComboBox()
        self.camera_combo.setMinimumWidth(120)
        self.detect_cameras_button = QPushButton("🔍 Detect Cameras")
        self.detect_cameras_button.setStyleSheet("background-color: teal; color: white;")
        self.detect_cameras_button.setToolTip("Scan for available cameras")
        self.detect_cameras_button.clicked.connect(self.detect_available_cameras)
        
        camera_layout = QHBoxLayout()
        camera_layout.addWidget(QLabel("Camera:"))
        camera_layout.addWidget(self.camera_combo)
        camera_layout.addWidget(self.detect_cameras_button)
        controls_layout.addLayout(camera_layout)
        
        self.browse_button = QPushButton("Browse Video")
        self.browse_button.setEnabled(False)
        controls_layout.addWidget(self.browse_button)
        
        # Load first frame button for polygon drawing
        self.load_frame_button = QPushButton("Load First Frame")
        self.load_frame_button.setStyleSheet("background-color: teal; color: white;")
        self.load_frame_button.setEnabled(False)
        self.load_frame_button.clicked.connect(self.load_first_frame)
        controls_layout.addWidget(self.load_frame_button)
        
        # Video playback controls
        playback_layout = QHBoxLayout()
        
        self.backward_button = QPushButton("⏪ -10s")
        self.backward_button.setStyleSheet("background-color: purple; color: white;")
        self.backward_button.setEnabled(False)
        self.backward_button.clicked.connect(self.seek_backward)
        playback_layout.addWidget(self.backward_button)
        
        self.forward_button = QPushButton("⏩ +10s") 
        self.forward_button.setStyleSheet("background-color: purple; color: white;")
        self.forward_button.setEnabled(False)
        self.forward_button.clicked.connect(self.seek_forward)
        playback_layout.addWidget(self.forward_button)
        
        # Speed control
        playback_layout.addWidget(QLabel("Speed:"))
        self.speed_combo = QComboBox()
        self.speed_combo.addItem("0.5x")
        self.speed_combo.addItem("1.0x")
        self.speed_combo.addItem("1.5x")
        self.speed_combo.addItem("2.0x")
        self.speed_combo.addItem("3.0x")
        self.speed_combo.addItem("4.0x")
        self.speed_combo.addItem("5.0x")
        self.speed_combo.addItem("6.0x")
        self.speed_combo.setCurrentText("1.0x")
        self.speed_combo.currentTextChanged.connect(self.change_playback_speed)
        self.speed_combo.setToolTip("Change video playback speed (only works with video files)")
        playback_layout.addWidget(self.speed_combo)
        
        controls_layout.addLayout(playback_layout)
        
        self.start_button = QPushButton("Start")
        self.pause_button = QPushButton("Pause")
        self.stop_button = QPushButton("Stop")
        self.debug_button = QPushButton("Save Debug Frame")
        
        self.start_button.setStyleSheet("background-color: green; color: white;")
        self.pause_button.setStyleSheet("background-color: orange; color: white;")
        self.stop_button.setStyleSheet("background-color: red; color: white;")
        self.debug_button.setStyleSheet("background-color: blue; color: white;")
        
        # Set size policies for buttons
        for button in [self.start_button, self.pause_button, self.stop_button, self.debug_button]:
            button.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Fixed)
            button.setMinimumHeight(30)
        
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.debug_button.setEnabled(False)
        
        controls_layout.addWidget(self.start_button)
        controls_layout.addWidget(self.pause_button)
        controls_layout.addWidget(self.stop_button)
        controls_layout.addWidget(self.debug_button)
        controls_layout.addStretch()  # Add stretch to push buttons together
        
        left_panel.addLayout(controls_layout)
        
        # Right panel - Information and settings
        right_panel_widget = QWidget()
        right_panel_widget.setMinimumWidth(300)  # Minimum width for right panel
        right_panel_widget.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        
        # Create scroll area for right panel content
        from PyQt5.QtWidgets import QScrollArea
        scroll_area = QScrollArea()
        scroll_area.setWidget(right_panel_widget)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        
        right_panel = QVBoxLayout(right_panel_widget)
        
        # Statistics group
        stats_group = QGroupBox("Statistics")
        stats_layout = QVBoxLayout(stats_group)
        
        self.stats_labels = {
            'frames': QLabel("Frames processed: 0"),
            'detections': QLabel("Total detections: 0"),
            'unique': QLabel("Unique plates: 0"),
            'recent': QLabel("Recent detections: 0"),
            'tracked': QLabel("Tracked cars: 0"),
            'parked': QLabel("Parked cars: 0"),
            'white_plates': QLabel("⚪ White plates: 0"),
            'green_plates': QLabel("🟢 Green plates: 0"),
            'red_plates': QLabel("🔴 Red plates: 0")
        }
        
        for label in self.stats_labels.values():
            label.setFont(QFont("Arial", 10))
            stats_layout.addWidget(label)
        
        right_panel.addWidget(stats_group)
        
        # Detection results
        results_group = QGroupBox("Recent Detections - Live Feed")
        results_layout = QVBoxLayout(results_group)
        
        # Add clear button for detections
        clear_button = QPushButton("Clear Detections")
        clear_button.setStyleSheet("background-color: orange; color: white;")
        clear_button.clicked.connect(self.clear_detections)
        results_layout.addWidget(clear_button)
        
        self.detections_text = QTextEdit()
        self.detections_text.setMaximumHeight(200)
        self.detections_text.setReadOnly(True)
        self.detections_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Courier New', monospace;
                font-size: 12px;
                font-weight: bold;
                background-color: #f0f0f0;
                border: 2px solid #4CAF50;
                line-height: 1.4;
                padding: 5px;
            }
        """)
        results_layout.addWidget(self.detections_text)
        
        right_panel.addWidget(results_group)
        
        # Entry/Exit Log
        events_group = QGroupBox("Vehicle Entry/Exit Log")
        events_layout = QVBoxLayout(events_group)
        self.events_text = QTextEdit()
        self.events_text.setReadOnly(True)
        self.events_text.setStyleSheet("""
            QTextEdit {
                font-family: 'Consolas', 'Courier New', monospace;
                font-size: 11px;
                background-color: #1e1e1e;
                color: #d4d4d4;
                border: 1px solid #333;
            }
        """)
        events_layout.addWidget(self.events_text)
        right_panel.addWidget(events_group)
        
        # Settings group
        settings_group = QGroupBox("Settings")
        settings_layout = QVBoxLayout(settings_group)
        
        # Confidence threshold
        conf_layout = QHBoxLayout()
        conf_layout.addWidget(QLabel("Confidence Threshold:"))
        self.conf_slider = QSlider(Qt.Horizontal)
        self.conf_slider.setMinimum(10)
        self.conf_slider.setMaximum(95)
        self.conf_slider.setValue(int(Config.CONFIDENCE_THRESHOLD * 100))
        self.conf_label = QLabel(f"{Config.CONFIDENCE_THRESHOLD:.2f}")
        conf_layout.addWidget(self.conf_slider)
        conf_layout.addWidget(self.conf_label)
        settings_layout.addLayout(conf_layout)
        
        # OCR confidence threshold
        ocr_conf_layout = QHBoxLayout()
        ocr_conf_layout.addWidget(QLabel("OCR Confidence:"))
        self.ocr_conf_slider = QSlider(Qt.Horizontal)
        self.ocr_conf_slider.setMinimum(10)
        self.ocr_conf_slider.setMaximum(95)
        self.ocr_conf_slider.setValue(int(Config.MIN_OCR_CONFIDENCE * 100))
        self.ocr_conf_label = QLabel(f"{Config.MIN_OCR_CONFIDENCE:.2f}")
        ocr_conf_layout.addWidget(self.ocr_conf_slider)
        ocr_conf_layout.addWidget(self.ocr_conf_label)
        settings_layout.addLayout(ocr_conf_layout)
        
        # Duplicate time window
        dup_layout = QHBoxLayout()
        dup_layout.addWidget(QLabel("Duplicate Window (s):"))
        self.dup_spinbox = QSpinBox()
        self.dup_spinbox.setMinimum(1)
        self.dup_spinbox.setMaximum(30)
        self.dup_spinbox.setValue(int(Config.DUPLICATE_TIME_WINDOW))
        dup_layout.addWidget(self.dup_spinbox)
        settings_layout.addLayout(dup_layout)
        
        # Frame processing control
        frame_layout = QHBoxLayout()
        frame_layout.addWidget(QLabel("Process Every N Frames:"))
        self.frame_skip_spinbox = QSpinBox()
        self.frame_skip_spinbox.setMinimum(1)
        self.frame_skip_spinbox.setMaximum(10)
        self.frame_skip_spinbox.setValue(3)  # Default to every 3rd frame
        self.frame_skip_spinbox.setToolTip("Higher values = faster processing, lower accuracy")
        frame_layout.addWidget(self.frame_skip_spinbox)
        settings_layout.addLayout(frame_layout)
        
        # Vehicle-Plate Association
        association_layout = QHBoxLayout()
        self.vehicle_plate_association = QCheckBox("Vehicle-Plate Association")
        self.vehicle_plate_association.setChecked(True)  # Enable by default
        self.vehicle_plate_association.setToolTip("Only detect plates near vehicles (reduces false positives)")
        association_layout.addWidget(self.vehicle_plate_association)
        settings_layout.addLayout(association_layout)
        
        # Save detections checkbox
        self.save_detections_checkbox = QCheckBox("Save Detections")
        self.save_detections_checkbox.setChecked(Config.SAVE_DETECTIONS)
        self.save_detections_checkbox.toggled.connect(self.update_save_detections_status)
        settings_layout.addWidget(self.save_detections_checkbox)
        
        # Camera configuration
        camera_config_group = QGroupBox("Camera Configuration")
        camera_config_layout = QVBoxLayout(camera_config_group)
        
        # Resolution control
        resolution_layout = QHBoxLayout()
        resolution_layout.addWidget(QLabel("Resolution:"))
        self.resolution_combo = QComboBox()
        self.resolution_combo.addItem("640x480")
        self.resolution_combo.addItem("1280x720 (HD)")
        self.resolution_combo.addItem("1920x1080 (FHD)")
        self.resolution_combo.addItem("Custom")
        self.resolution_combo.setCurrentText("1280x720 (HD)")
        resolution_layout.addWidget(self.resolution_combo)
        camera_config_layout.addLayout(resolution_layout)
        
        # Custom resolution
        custom_res_layout = QHBoxLayout()
        custom_res_layout.addWidget(QLabel("Custom Width:"))
        self.custom_width_spinbox = QSpinBox()
        self.custom_width_spinbox.setMinimum(320)
        self.custom_width_spinbox.setMaximum(3840)
        self.custom_width_spinbox.setValue(1280)
        custom_res_layout.addWidget(self.custom_width_spinbox)
        
        custom_res_layout.addWidget(QLabel("Height:"))
        self.custom_height_spinbox = QSpinBox()
        self.custom_height_spinbox.setMinimum(240)
        self.custom_height_spinbox.setMaximum(2160)
        self.custom_height_spinbox.setValue(720)
        custom_res_layout.addWidget(self.custom_height_spinbox)
        camera_config_layout.addLayout(custom_res_layout)
        
        # Camera FPS
        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("Camera FPS:"))
        self.camera_fps_spinbox = QSpinBox()
        self.camera_fps_spinbox.setMinimum(1)
        self.camera_fps_spinbox.setMaximum(120)
        self.camera_fps_spinbox.setValue(30)
        fps_layout.addWidget(self.camera_fps_spinbox)
        camera_config_layout.addLayout(fps_layout)
        
        settings_layout.addWidget(camera_config_group)
        
        # Detection mode selection
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Detection Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Real-time Detection")
        self.mode_combo.addItem("Parking Vehicle Detection")
        self.mode_combo.addItem("Zone-based Detection")
        self.mode_combo.setToolTip("Select detection mode")
        mode_layout.addWidget(self.mode_combo)
        settings_layout.addLayout(mode_layout)
        
        # Parking time threshold (only for parking mode)
        parking_time_layout = QHBoxLayout()
        parking_time_layout.addWidget(QLabel("Parking Time (s):"))
        self.parking_time_spinbox = QSpinBox()
        self.parking_time_spinbox.setMinimum(1)
        self.parking_time_spinbox.setMaximum(60)
        self.parking_time_spinbox.setValue(3)  # Default 3 seconds
        self.parking_time_spinbox.setToolTip("Minimum time a car must remain stationary to be detected as parked")
        parking_time_layout.addWidget(self.parking_time_spinbox)
        settings_layout.addLayout(parking_time_layout)
        
        # Point controls group
        point_group = QGroupBox("Detection Zone Controls")
        point_layout = QVBoxLayout(point_group)
        
        point_buttons_layout = QHBoxLayout()
        self.point_button = QPushButton("Draw Detection Zone")
        self.point_button.setStyleSheet("background-color: purple; color: white;")
        self.point_button.setEnabled(False)
        self.point_button.clicked.connect(self.toggle_point_mode)
        self.point_button.setToolTip("Draw two parallel lines to define a detection zone")
        point_buttons_layout.addWidget(self.point_button)
        
        self.clear_point_button = QPushButton("Clear Zone")
        self.clear_point_button.setStyleSheet("background-color: brown; color: white;")
        self.clear_point_button.setEnabled(False)
        self.clear_point_button.clicked.connect(self.clear_points)
        point_buttons_layout.addWidget(self.clear_point_button)
        
        point_layout.addLayout(point_buttons_layout)
        
        # Detection points display
        self.point_display_text = QTextEdit()
        self.point_display_text.setMaximumHeight(100)
        self.point_display_text.setReadOnly(True)
        self.point_display_text.setPlaceholderText("Detection zone will appear here...")
        self.point_display_text.setStyleSheet("font-family: monospace; font-size: 12px;")
        point_layout.addWidget(self.point_display_text)
        
        settings_layout.addWidget(point_group)
        
        # Video seek control
        seek_layout = QHBoxLayout()
        seek_layout.addWidget(QLabel("Start from Frame:"))
        self.seek_spinbox = QSpinBox()
        self.seek_spinbox.setMinimum(0)
        self.seek_spinbox.setMaximum(999999)
        self.seek_spinbox.setValue(0)
        self.seek_spinbox.setToolTip("Frame number to start detection from")
        seek_layout.addWidget(self.seek_spinbox)
        settings_layout.addLayout(seek_layout)
        
        right_panel.addWidget(settings_group)
        
        # Model status
        model_group = QGroupBox("Model Status")
        model_layout = QVBoxLayout(model_group)
        
        self.yolo_status = QLabel("YOLO: Not loaded")
        self.crnn_status = QLabel("CRNN: Not loaded")
        self.camera_status = QLabel("Camera: Ready")
        
        model_layout.addWidget(self.yolo_status)
        model_layout.addWidget(self.crnn_status)
        model_layout.addWidget(self.camera_status)
        
        right_panel.addWidget(model_group)
        
        # Add stretch to push everything up
        right_panel.addStretch()
        
        # Add panels to splitter for draggable resizing
        left_widget = QWidget()
        left_widget.setLayout(left_panel)
        left_widget.setMinimumWidth(400)  # Minimum width for video area
        left_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        
        self.main_splitter.addWidget(left_widget)
        self.main_splitter.addWidget(scroll_area)
        
        # Set initial sizes: ~75% for video, ~25% for right panel
        self.main_splitter.setSizes([1050, 350])
        self.main_splitter.setCollapsible(0, False)  # Don't allow collapsing video panel
        self.main_splitter.setCollapsible(1, False)  # Don't allow collapsing settings panel
        
    def setup_connections(self):
        """Setup signal connections"""
        self.source_combo.currentTextChanged.connect(self.on_source_changed)
        self.browse_button.clicked.connect(self.browse_video_file)
        
        self.start_button.clicked.connect(self.start_detection)
        self.pause_button.clicked.connect(self.pause_detection)
        self.stop_button.clicked.connect(self.stop_detection)
        self.debug_button.clicked.connect(self.save_debug_frame)
        
        self.conf_slider.valueChanged.connect(self.update_conf_threshold)
        self.ocr_conf_slider.valueChanged.connect(self.update_ocr_conf_threshold)
        self.dup_spinbox.valueChanged.connect(self.update_duplicate_window)
        self.frame_skip_spinbox.valueChanged.connect(self.update_frame_skip)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        self.parking_time_spinbox.valueChanged.connect(self.update_parking_time_threshold)
        self.vehicle_plate_association.toggled.connect(self.update_vehicle_plate_association)
        self.save_detections_checkbox.toggled.connect(self.update_save_detections_status)
    
    def seek_backward(self):
        """Seek backward 10 seconds in video"""
        if self.video_thread and self.source_combo.currentText() == 'Video File':
            self.video_thread.request_seek(-10)
    
    def seek_forward(self):
        """Seek forward 10 seconds in video"""
        if self.video_thread and self.source_combo.currentText() == 'Video File':
            self.video_thread.request_seek(10)
    
    def change_playback_speed(self, speed_text):
        """Change video playback speed"""
        try:
            speed_multiplier = float(speed_text.replace('x', ''))
            if self.video_thread and self.video_thread.isRunning():
                self.video_thread.playback_speed = speed_multiplier
                logger.info(f"Playback speed changed to {speed_multiplier}x")
                
                # Update the speed immediately if video is running
                if hasattr(self.video_thread, 'running') and self.video_thread.running:
                    logger.info(f"✅ Speed control active: {speed_multiplier}x")
            else:
                logger.info(f"Speed will be set to {speed_multiplier}x when video starts")
        except Exception as e:
            logger.error(f"Error changing playback speed: {e}")
    
    def on_source_changed(self, source_type):
        """Handle source type change"""
        is_video = source_type == "Video File"
        is_camera = source_type == "Camera"
        
        # Video-specific controls
        self.browse_button.setEnabled(is_video)
        self.load_frame_button.setEnabled(is_video)
        self.backward_button.setEnabled(False)  # Will be enabled when video starts
        self.forward_button.setEnabled(False)
        self.speed_combo.setEnabled(is_video)
        self.point_button.setEnabled(False)  # Will be enabled after loading frame
        
        # Camera-specific controls
        self.camera_combo.setEnabled(is_camera)
        self.detect_cameras_button.setEnabled(is_camera)
        
        if not is_video:
            self.clear_points()  # Clear points when switching to camera
        
    def browse_video_file(self):
        """Browse for video file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select Video File", str(Config.VIDEO_PATH),
            "Video Files (*.mp4 *.avi *.mov *.mkv *.wmv);;All Files (*)"
        )
        
        if file_path:
            self.video_file_path = file_path
            self.browse_button.setText(f"Selected: {Path(file_path).name}")
            
            # Enable load frame button
            self.load_frame_button.setEnabled(True)
            
            # Load video info
            cap = cv2.VideoCapture(file_path)
            if cap.isOpened():
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                self.seek_spinbox.setMaximum(total_frames - 1)
                fps = cap.get(cv2.CAP_PROP_FPS)
                logger.info(f"Video loaded: {total_frames} frames, {fps:.2f} FPS")
                cap.release()
    
    def start_detection(self):
        """Start detection process"""
        try:
            # Clear previous detection history when restarting
            self.clear_detections()
            
            source_type = 'camera' if self.source_combo.currentText() == 'Camera' else 'video'
            source_path = None
            
            if source_type == 'video':
                if hasattr(self, 'video_file_path'):
                    source_path = self.video_file_path
                else:
                    # Use default video path
                    video_files = list(Path(Config.VIDEO_PATH).glob("*.mp4"))
                    if video_files:
                        source_path = str(video_files[0])
                    else:
                        logger.error("No video file selected or found")
                        return
            
            # Get start frame and detection settings based on mode
            start_frame = self.seek_spinbox.value()
            detection_mode = self.mode_combo.currentText()
            
            # Configure detection settings based on mode
            detection_point = None
            parking_enabled = False
            
            if detection_mode == "Zone-based Detection":
                detection_point = self.detection_points if self.points_complete else None
                if not detection_point or len(detection_point) != 2:
                    logger.warning("Zone-based detection selected but detection line not complete!")
            elif detection_mode == "Parking Vehicle Detection":
                parking_enabled = True
            # Real-time Detection uses default settings
            
            parking_time_threshold = self.parking_time_spinbox.value()
            
            # Get camera parameters if using camera
            camera_index = 0
            camera_config = {}
            if source_type == 'camera':
                camera_index = self.get_selected_camera_index()
                camera_config = self.get_camera_config()
                logger.info(f"Using camera {camera_index} with config: {camera_config}")
            
            # Create and start video thread
            self.video_thread = VideoThread(source_type, source_path, start_frame, detection_point,
                                          parking_enabled, parking_time_threshold, camera_index, camera_config)
            self.video_thread.frame_ready.connect(self.update_video_display)
            self.video_thread.stats_updated.connect(self.update_stats)
            self.video_thread.models_loaded.connect(self.update_model_status)
            
            # Set playback speed from UI
            speed_multiplier = float(self.speed_combo.currentText().replace('x', ''))
            self.video_thread.playback_speed = speed_multiplier
            

            
            self.video_thread.start()
            
            # Start live feed timer
            self.live_feed_timer.start()
            
            # Update UI
            self.start_button.setEnabled(False)
            self.pause_button.setEnabled(True)
            self.stop_button.setEnabled(True)
            self.debug_button.setEnabled(True)
            
            # Enable video controls for video files
            if source_type == 'video':
                self.backward_button.setEnabled(True)
                self.forward_button.setEnabled(True)
            
            # Update camera status
            if source_type == 'camera':
                camera_name = f"Camera {camera_index}"
                if self.available_cameras:
                    for cam in self.available_cameras:
                        if cam['index'] == camera_index:
                            camera_name = f"Camera {camera_index} ({cam['resolution']})"
                            break
                self.camera_status.setText(f"Camera: {camera_name} Active")
                self.camera_status.setStyleSheet("color: green;")
            else:
                self.camera_status.setText("Camera: Not used")
                self.camera_status.setStyleSheet("color: gray;")
            
            logger.info(f"Started detection with {source_type} source - Mode: {detection_mode} - Live feed active")
            
        except Exception as e:
            logger.error(f"Error starting detection: {e}")
    
    def pause_detection(self):
        """Pause/resume detection"""
        if self.video_thread:
            if self.video_thread.paused:
                self.video_thread.resume()
                self.pause_button.setText("Pause")
            else:
                self.video_thread.pause()
                self.pause_button.setText("Resume")
    
    def stop_detection(self):
        """Stop detection process"""
        if self.video_thread:
            self.video_thread.stop()
            self.video_thread = None
        
        # Stop live feed timer
        self.live_feed_timer.stop()
        
        # Update UI
        self.start_button.setEnabled(True)
        self.pause_button.setEnabled(False)
        self.stop_button.setEnabled(False)
        self.debug_button.setEnabled(False)
        self.pause_button.setText("Pause")
        
        # Disable video controls
        self.backward_button.setEnabled(False)
        self.forward_button.setEnabled(False)
        
        # Reset load frame button
        self.load_frame_button.setText("Load First Frame")
        self.load_frame_button.setStyleSheet("background-color: teal; color: white;")
        self.first_frame_loaded = False
        
        # Reset camera status
        self.camera_status.setText("Camera: Ready")
        self.camera_status.setStyleSheet("color: black;")
    
    def update_video_display(self, frame, detections):
        """Update video display with frame and detections"""
        try:
            # Store current frame for debugging
            self.current_frame = frame.copy()
            
            # Add ALL new detections to history for live feed (including hidden ones for counting)
            for detection in detections:
                if detection.detection_type == "plate" and detection.text not in ["Car", "Plate", "Car (Debug)", "Car (Near Line)", "Car (Crossed)"]:
                    # Only add if it's not already in history (avoid duplicates)
                    if not any(d.text == detection.text and abs(d.timestamp - detection.timestamp) < 1.0 for d in self.all_detections_history[-10:]):
                        self.all_detections_history.append(detection)
                        logger.info(f"Added to live feed history: {detection.text}")
                        
                        # Also immediately update the live feed to show new detection
                        self.update_live_feed()
                
                # Add entry/exit events to their own history
                if detection.detection_type in ["plate", "exit"]:
                    self.all_events_history.append(detection)
                    logger.info(f"Added to event log: {detection.detection_type} - {detection.text}")
            
            # Draw detection boxes and labels
            display_frame = frame.copy()
            
            # Draw detection zone if active
            if self.points_complete and len(self.detection_points) == 4:
                # Define polygon for visualization
                p_a, p_b = self.detection_points[0], self.detection_points[1]
                p_c, p_d = self.detection_points[2], self.detection_points[3]
                zone_polygon = np.array([p_a, p_b, p_d, p_c], dtype=np.int32)
                
                # Draw the transparent polygon zone
                overlay = display_frame.copy()
                cv2.fillPoly(overlay, [zone_polygon], (0, 255, 100)) # Light green fill
                alpha = 0.2  # Transparency factor
                display_frame = cv2.addWeighted(overlay, alpha, display_frame, 1 - alpha, 0)

                # Draw Entry Line (Green)
                cv2.line(display_frame, self.detection_points[0], self.detection_points[1], (0, 255, 0), 3)
                # Draw Exit Line (Red)
                cv2.line(display_frame, self.detection_points[2], self.detection_points[3], (0, 0, 255), 3)

            
            for detection in detections:
                x1, y1, x2, y2 = detection.bbox
                
                # Choose colors based on detection type
                if detection.detection_type == "car":
                    box_color = (255, 0, 0)  # Blue for cars
                    label = "Car"
                    if "In-Zone" in detection.text:
                        label = "Car (In-Zone)"
                        box_color = (255, 200, 0) # Light Blue for in-zone cars
                    show_confidence = False
                elif detection.detection_type == "parked_car":
                    box_color = (0, 165, 255)  # Orange for parked cars
                    label = detection.text  # Already formatted with duration
                    show_confidence = False
                elif detection.detection_type == "plate" and detection.text != "Car":
                    box_color = (0, 255, 0)  # Green for recognized plates
                    # Add plate type emoji and info
                    type_emoji = self.anpr_processor.get_plate_type_emoji(detection.plate_type) if hasattr(self, 'anpr_processor') else ""
                    type_text = f"[{detection.plate_type.upper()}]" if detection.plate_type != "unknown" else ""
                    
                    # Add crossed line indicator for zone-based detections
                    if hasattr(self, 'video_thread') and self.video_thread and self.video_thread.detection_point:
                        label = f"ZONE: {type_emoji}{detection.text} {type_text} ({detection.confidence:.2f})"
                    else:
                        label = f"{type_emoji}{detection.text} {type_text} ({detection.confidence:.2f})"
                    show_confidence = True
                elif detection.detection_type == "plate_detected":
                    box_color = (0, 255, 255)  # Yellow for detected but not processed plates
                    label = "License Plate"
                    show_confidence = False
                else:
                    continue  # Skip other types
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)
                
                # Draw label
                if show_confidence or detection.detection_type == "car":
                    font_scale = 0.5
                    label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 2)[0]
                    
                    # Background for text
                    cv2.rectangle(display_frame, (x1, y1 - label_size[1] - 10), 
                                 (x1 + label_size[0], y1), box_color, -1)
                    
                    # Text
                    cv2.putText(display_frame, label, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 2)
            
            # Convert to QImage and display
            height, width, channel = display_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # Scale to fit label while maintaining aspect ratio
            label_size = self.video_label.size()
            if label_size.width() > 0 and label_size.height() > 0:
                scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                    label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
                )
                self.video_label.setPixmap(scaled_pixmap)
            else:
                # Fallback if label size is not valid
                self.video_label.setPixmap(QPixmap.fromImage(q_image))
            
            # Update current detections for other uses
            self.current_detections = detections
            
        except Exception as e:
            logger.error(f"Error updating video display: {e}")
    
    def update_stats(self, stats):
        """Update statistics display"""
        try:
            self.stats_labels['frames'].setText(f"Frames processed: {stats['frame_count']}")
            self.stats_labels['detections'].setText(f"Total detections: {stats['total_detections']}")
            self.stats_labels['unique'].setText(f"Unique plates: {stats['unique_plates']}")
            self.stats_labels['recent'].setText(f"Recent detections: {stats['recent_detections']}")
            
            if 'tracked_cars' in stats:
                self.stats_labels['tracked'].setText(f"Tracked cars: {stats['tracked_cars']}")
            if 'parked_cars' in stats:
                self.stats_labels['parked'].setText(f"Parked cars: {stats['parked_cars']}")
            
            # Update plate type statistics
            if 'white_plates' in stats:
                self.stats_labels['white_plates'].setText(f"⚪ White plates: {stats['white_plates']}")
            if 'green_plates' in stats:
                self.stats_labels['green_plates'].setText(f"🟢 Green plates: {stats['green_plates']}")
            if 'red_plates' in stats:
                self.stats_labels['red_plates'].setText(f"🔴 Red plates: {stats['red_plates']}")
            
        except Exception as e:
            logger.error(f"Error updating stats: {e}")
    
    def update_conf_threshold(self, value):
        """Update confidence threshold"""
        Config.CONFIDENCE_THRESHOLD = value / 100.0
        self.conf_label.setText(f"{Config.CONFIDENCE_THRESHOLD:.2f}")
    
    def update_ocr_conf_threshold(self, value):
        """Update OCR confidence threshold"""
        Config.MIN_OCR_CONFIDENCE = value / 100.0
        self.ocr_conf_label.setText(f"{Config.MIN_OCR_CONFIDENCE:.2f}")
    
    def update_duplicate_window(self, value):
        """Update duplicate detection time window"""
        Config.DUPLICATE_TIME_WINDOW = float(value)
    
    def update_frame_skip(self, value):
        """Update frame skip setting"""
        if self.video_thread and self.video_thread.anpr_processor:
            self.video_thread.anpr_processor.process_every_n_frames = value
            logger.info(f"Frame skip updated to every {value} frames")
    
    def on_mode_changed(self, mode):
        """Handle detection mode change"""
        logger.info(f"Detection mode changed to: {mode}")
        
        # Update UI based on mode
        if mode == "Zone-based Detection":
            self.point_button.setEnabled(hasattr(self, 'first_frame_loaded') and self.first_frame_loaded)
            self.clear_point_button.setEnabled(self.points_complete)
        else:
            self.point_button.setEnabled(False)
            self.clear_point_button.setEnabled(False)
    
    def load_first_frame(self):
        """Load first frame of video for polygon drawing"""
        try:
            if hasattr(self, 'video_file_path'):
                cap = cv2.VideoCapture(self.video_file_path)
                if cap.isOpened():
                    # Get frame from seek position
                    start_frame = self.seek_spinbox.value()
                    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                    
                    ret, frame = cap.read()
                    if ret:
                        self.current_frame_for_point = frame.copy()
                        self.current_frame = frame.copy()  # For display
                        self.first_frame_loaded = True
                        
                        # Display the frame
                        self.display_frame_with_point()
                        
                        # Enable point selection for zone-based mode
                        if self.mode_combo.currentText() == "Zone-based Detection":
                            self.point_button.setEnabled(True)
                        
                        self.load_frame_button.setText("Frame Loaded ✓")
                        self.load_frame_button.setStyleSheet("background-color: green; color: white;")
                        
                        logger.info(f"First frame loaded successfully from frame {start_frame}")
                    else:
                        logger.error("Failed to read frame from video")
                    
                    cap.release()
                else:
                    logger.error("Failed to open video file")
            else:
                logger.error("No video file selected")
        except Exception as e:
            logger.error(f"Error loading first frame: {e}")
    
    def update_point_display(self):
        """Update the detection points display in the GUI"""
        num_points = len(self.detection_points)
        text = "Detection Zone Status:\n\n"
        
        if num_points == 0:
            text += "🔵 Click 'Draw Zone' then click 4 points on the video frame.\n\n"
            text += "Line 1 (Entry): Point A -> Point B\n"
            text += "Line 2 (Exit):  Point C -> Point D"
        elif num_points == 1:
            p_a = self.detection_points[0]
            text += f"✅ Entry Line Start (A): {p_a}\n"
            text += "🔵 Click Point B to finish Entry Line..."
        elif num_points == 2:
            p_a, p_b = self.detection_points
            text += f"✅ Entry Line (A->B): {p_a} to {p_b}\n"
            text += "🔵 Click Point C to start Exit Line..."
        elif num_points == 3:
            p_a, p_b = self.detection_points[:2]
            p_c = self.detection_points[2]
            text += f"✅ Entry Line (A->B): {p_a} to {p_b}\n"
            text += f"✅ Exit Line Start (C): {p_c}\n"
            text += "🔵 Click Point D to finish Exit Line..."
        elif num_points == 4:
            p_a, p_b = self.detection_points[:2]
            p_c, p_d = self.detection_points[2:]
            text += f"✅ Entry Line (A->B): {p_a} to {p_b}\n"
            text += f"✅ Exit Line (C->D): {p_c} to {p_d}\n\n"
            text += "✅ Zone is ready for detection."
            
        self.point_display_text.setPlainText(text)
    
    def update_parking_time_threshold(self, value):
        """Update parking time threshold"""
        if self.video_thread and hasattr(self.video_thread, 'anpr_processor'):
            self.video_thread.anpr_processor.parking_time_threshold = float(value)
            logger.info(f"Parking time threshold updated to {value} seconds")
    
    def update_vehicle_plate_association(self, enabled):
        """Update vehicle-plate association filtering"""
        if self.video_thread and hasattr(self.video_thread, 'anpr_processor'):
            self.video_thread.anpr_processor.vehicle_plate_association_enabled = enabled
            status = "enabled" if enabled else "disabled"
            logger.info(f"Vehicle-plate association filtering {status}")
    
    def toggle_point_mode(self):
        """Toggle point selection mode"""
        if not self.selecting_points:
            # Start fresh if zone was already complete
            if self.points_complete:
                self.clear_points()

            self.selecting_points = True
            self.point_button.setText("Cancel Drawing")
            self.point_button.setStyleSheet("background-color: red; color: white;")
            self.clear_point_button.setEnabled(False)
            
            logger.info("Point selection mode activated - Click 4 points to draw detection zone")
            self.instructions_label.setText("🔵 Draw Entry Line (A->B) then Exit Line (C->D)")
            self.instructions_label.show()
        else:
            self.cancel_point_selection()
    
    def cancel_point_selection(self):
        """Cancel point selection"""
        self.selecting_points = False
        self.point_button.setText("Draw Detection Zone")
        self.point_button.setStyleSheet("background-color: purple; color: white;")
        self.clear_point_button.setEnabled(self.points_complete)
        self.instructions_label.hide()
        logger.info("Point selection cancelled")
    
    def clear_points(self):
        """Clear the selected points"""
        self.detection_points.clear()
        self.points_complete = False
        self.selecting_points = False
        self.point_button.setText("Draw Detection Zone")
        self.point_button.setStyleSheet("background-color: purple; color: white;")
        self.clear_point_button.setEnabled(False)
        
        # Redisplay current frame without points
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            self.display_frame_with_point()
        
        # Update point display
        self.update_point_display()
        
        logger.info("Detection line cleared")
    
    def mouse_press_event(self, event):
        """Handle mouse press events for point selection"""
        if not self.selecting_points or len(self.detection_points) >= 4:
            return
        
        # Get the position relative to the video label
        pos = event.pos()
        
        # Convert position to frame coordinates
        if hasattr(self, 'current_frame_for_point') and self.current_frame_for_point is not None:
            frame_height, frame_width = self.current_frame_for_point.shape[:2]
            label_width = self.video_label.width()
            label_height = self.video_label.height()
            
            # Get the actual pixmap size to handle aspect ratio correctly
            pixmap = self.video_label.pixmap()
            if pixmap:
                pixmap_width = pixmap.width()
                pixmap_height = pixmap.height()
                
                # Calculate the actual display area within the label (considering aspect ratio)
                aspect_ratio = frame_width / frame_height
                label_aspect_ratio = label_width / label_height
                
                if aspect_ratio > label_aspect_ratio:
                    # Image is wider, fit to width
                    display_width = label_width
                    display_height = int(label_width / aspect_ratio)
                    offset_x = 0
                    offset_y = (label_height - display_height) // 2
                else:
                    # Image is taller, fit to height
                    display_width = int(label_height * aspect_ratio)
                    display_height = label_height
                    offset_x = (label_width - display_width) // 2
                    offset_y = 0
                
                # Adjust click position
                adjusted_x = pos.x() - offset_x
                adjusted_y = pos.y() - offset_y
                
                # Only process if click is within the actual image area
                if 0 <= adjusted_x <= display_width and 0 <= adjusted_y <= display_height:
                    # Calculate scaling factors
                    scale_x = frame_width / display_width
                    scale_y = frame_height / display_height
                    
                    # Convert to frame coordinates
                    frame_x = int(adjusted_x * scale_x)
                    frame_y = int(adjusted_y * scale_y)
                else:
                    return  # Click was outside the image area
            else:
                # Fallback to old method if no pixmap
                scale_x = frame_width / label_width if label_width > 0 else 1
                scale_y = frame_height / label_height if label_height > 0 else 1
                frame_x = int(pos.x() * scale_x)
                frame_y = int(pos.y() * scale_y)
            
            if event.button() == 1:  # Left click - add point
                self.detection_points.append((frame_x, frame_y))
                logger.info(f"Detection point {len(self.detection_points)} added: ({frame_x}, {frame_y})")
                
                if len(self.detection_points) == 4:
                    # Four points selected, complete the zone
                    self.points_complete = True
                    self.cancel_point_selection()
                    self.clear_point_button.setEnabled(True)
                    logger.info("Detection zone completed")
                
                self.display_frame_with_point()
                self.update_point_display()
    
    def display_frame_with_point(self):
        """Display current frame with detection zone overlay"""
        if not hasattr(self, 'current_frame_for_point') or self.current_frame_for_point is None:
            return
        
        display_frame = self.current_frame_for_point.copy()
        
        # Draw detection points and lines based on how many points exist
        num_points = len(self.detection_points)
        
        # Draw first line (Entry)
        if num_points >= 1:
            x1, y1 = self.detection_points[0]
            cv2.circle(display_frame, (x1, y1), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, "A (Entry)", (x1 + 15, y1 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        if num_points >= 2:
            x2, y2 = self.detection_points[1]
            cv2.circle(display_frame, (x2, y2), 8, (0, 255, 0), -1)
            cv2.putText(display_frame, "B", (x2 + 15, y2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            # Draw Entry line
            x1, y1 = self.detection_points[0]
            cv2.line(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 3) # Green for entry
            
        # Draw second line (Exit)
        if num_points >= 3:
            x3, y3 = self.detection_points[2]
            cv2.circle(display_frame, (x3, y3), 8, (0, 0, 255), -1)
            cv2.putText(display_frame, "C (Exit)", (x3 + 15, y3 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        if num_points >= 4:
            x4, y4 = self.detection_points[3]
            cv2.circle(display_frame, (x4, y4), 8, (0, 0, 255), -1)
            cv2.putText(display_frame, "D", (x4 + 15, y4 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            # Draw Exit line
            x3, y3 = self.detection_points[2]
            cv2.line(display_frame, (x3, y3), (x4, y4), (0, 0, 255), 3) # Red for exit
        
        # Convert and display
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        
        label_size = self.video_label.size()
        if label_size.width() > 0 and label_size.height() > 0:
            scaled_pixmap = QPixmap.fromImage(q_image).scaled(
                label_size, Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.video_label.setPixmap(scaled_pixmap)
        else:
            # Fallback if label size is not valid
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
    
    def save_debug_frame(self):
        """Save current frame for debugging"""
        try:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                debug_path = f"debug_frame_{timestamp}.jpg"
                cv2.imwrite(debug_path, self.current_frame)
                logger.info(f"Debug frame saved: {debug_path}")
                
                # ENHANCED DEBUG: Force comprehensive analysis of this specific frame
                self._comprehensive_debug_analysis(self.current_frame, timestamp)
            else:
                logger.warning("No current frame available for debug")
        except Exception as e:
            logger.error(f"Error saving debug frame: {e}")
    
    def _comprehensive_debug_analysis(self, frame, timestamp):
        """Perform comprehensive debug analysis on a frame"""
        try:
            logger.info("🔍 === COMPREHENSIVE DEBUG ANALYSIS ===")
            
            if not self.video_thread or not self.video_thread.anpr_processor:
                logger.error("❌ Video thread or ANPR processor not available")
                return
                
            processor = self.video_thread.anpr_processor
            
            # 1. Test YOLO detection with very low confidence
            logger.info("🔍 Step 1: Testing YOLO detection...")
            if processor.yolo_model:
                with torch.no_grad():
                    results = processor.yolo_model(frame, conf=0.05, verbose=False, device='cpu')
                
                total_detections = 0
                vehicles = []
                plates = []
                
                for result in results:
                    if result.boxes is not None:
                        total_detections += len(result.boxes)
                        for box in result.boxes:
                            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                            conf = box.conf[0].cpu().numpy()
                            cls = int(box.cls[0].cpu().numpy()) if hasattr(box, 'cls') else -1
                            
                            logger.info(f"🔍 YOLO Detection: class={cls}, conf={conf:.3f}, bbox=({x1},{y1},{x2},{y2})")
                            
                            if cls == 0 or cls == 1:  # Vehicle
                                vehicles.append((x1, y1, x2, y2))
                            elif cls == 2:  # License plate
                                plates.append((x1, y1, x2, y2))
                
                logger.info(f"🔍 YOLO Results: {total_detections} total, {len(vehicles)} vehicles, {len(plates)} plates")
                
                # 2. Test OCR on all detected plates
                logger.info("🔍 Step 2: Testing OCR on detected plates...")
                for i, plate_bbox in enumerate(plates):
                    x1, y1, x2, y2 = plate_bbox
                    plate_region = frame[y1:y2, x1:x2]
                    
                    if plate_region.size > 0:
                        debug_plate_path = Config.OUTPUT_DIR / f"debug_comprehensive_plate_{i}_{timestamp}.jpg"
                        cv2.imwrite(str(debug_plate_path), plate_region)
                        logger.info(f"💾 Saved plate {i} to {debug_plate_path}")
                        
                        text, ocr_conf = processor.recognize_plate_text(plate_region)
                        logger.info(f"🔍 OCR Plate {i}: '{text}' (conf: {ocr_conf:.3f})")
                        
                        # Test plate type detection
                        plate_type = processor.detect_plate_type(plate_region)
                        logger.info(f"🎨 Plate {i} type: {plate_type}")
                
                # 3. If no plates detected, manually crop likely plate areas
                if len(plates) == 0:
                    logger.warning("⚠️ Step 3: No plates detected, attempting manual region extraction...")
                    self._manual_plate_region_test(frame, vehicles, timestamp, processor)
                    
            else:
                logger.error("❌ YOLO model not loaded")
                
            # 4. Test OCR model status
            logger.info("🔍 Step 4: Testing OCR model status...")
            if processor.crnn_model is not None:
                logger.info("✅ CRNN model is loaded")
                logger.info(f"🔍 Character set size: {len(processor.char_list) if processor.char_list else 'None'}")
            else:
                logger.error("❌ CRNN model is not loaded")
                
            logger.info("🔍 === DEBUG ANALYSIS COMPLETE ===")
            
        except Exception as e:
            logger.error(f"Error in comprehensive debug analysis: {e}")
    
    def _manual_plate_region_test(self, frame, vehicles, timestamp, processor):
        """Manually test potential plate regions when YOLO doesn't detect plates"""
        try:
            logger.info("🔍 Manual plate region extraction...")
            
            for i, vehicle_bbox in enumerate(vehicles):
                vx1, vy1, vx2, vy2 = vehicle_bbox
                vehicle_height = vy2 - vy1
                vehicle_width = vx2 - vx1
                
                # Try common plate locations in vehicles
                test_regions = [
                    # Front lower area
                    (vx1 + int(vehicle_width * 0.2), vy2 - int(vehicle_height * 0.3), 
                     vx1 + int(vehicle_width * 0.8), vy2 - int(vehicle_height * 0.1)),
                    # Front center
                    (vx1 + int(vehicle_width * 0.3), vy1 + int(vehicle_height * 0.7), 
                     vx1 + int(vehicle_width * 0.7), vy1 + int(vehicle_height * 0.9)),
                ]
                
                for j, (rx1, ry1, rx2, ry2) in enumerate(test_regions):
                    # Ensure coordinates are within frame bounds
                    rx1 = max(0, min(rx1, frame.shape[1]))
                    ry1 = max(0, min(ry1, frame.shape[0]))
                    rx2 = max(rx1, min(rx2, frame.shape[1]))
                    ry2 = max(ry1, min(ry2, frame.shape[0]))
                    
                    if rx2 > rx1 and ry2 > ry1:
                        test_region = frame[ry1:ry2, rx1:rx2]
                        
                        if test_region.size > 100:  # Minimum size check
                            manual_path = Config.OUTPUT_DIR / f"debug_manual_region_v{i}_r{j}_{timestamp}.jpg"
                            cv2.imwrite(str(manual_path), test_region)
                            
                            text, conf = processor.recognize_plate_text(test_region)
                            logger.info(f"🔍 Manual region V{i}R{j}: '{text}' (conf: {conf:.3f}) - {manual_path.name}")
                            
        except Exception as e:
            logger.error(f"Error in manual plate region test: {e}")
    
    def update_model_status(self, yolo_loaded, crnn_loaded):
        """Update model status indicators"""
        try:
            if yolo_loaded:
                self.yolo_status.setText("YOLO: ✓ Loaded")
                self.yolo_status.setStyleSheet("color: green;")
            else:
                self.yolo_status.setText("YOLO: ✗ Not loaded")
                self.yolo_status.setStyleSheet("color: red;")
            
            if crnn_loaded:
                self.crnn_status.setText("CRNN: ✓ Loaded")
                self.crnn_status.setStyleSheet("color: green;")
            else:
                self.crnn_status.setText("CRNN: ✗ Not loaded")
                self.crnn_status.setStyleSheet("color: red;")
                
        except Exception as e:
            logger.error(f"Error updating model status: {e}")
    
    def resizeEvent(self, event):
        """Handle window resize events"""
        super().resizeEvent(event)
        
        # Refresh video display to maintain proper scaling
        if hasattr(self, 'current_frame') and self.current_frame is not None:
            # Re-trigger video display update to adjust to new size
            QTimer.singleShot(50, self._refresh_video_display)  # Small delay to ensure layout is updated
    
    def _refresh_video_display(self):
        """Refresh video display with current frame"""
        try:
            if hasattr(self, 'current_frame') and self.current_frame is not None:
                # Check if we're in point selection mode
                if hasattr(self, 'current_frame_for_point') and self.current_frame_for_point is not None and hasattr(self, 'detection_points'):
                    self.display_frame_with_point()
                else:
                    # Re-display current frame with detections
                    if hasattr(self, 'current_detections'):
                        self.update_video_display(self.current_frame, self.current_detections)
        except Exception as e:
            logger.debug(f"Error refreshing video display: {e}")
    
    def closeEvent(self, event):
        """Handle application closing"""
        if self.video_thread:
            self.video_thread.stop()
        
        # Stop live feed timer
        self.live_feed_timer.stop()
        
        event.accept()
    
    def clear_detections(self):
        """Clear all detection history"""
        try:
            self.all_detections_history.clear()
            self.current_detections.clear()
            self.detections_text.clear()
            self.all_events_history.clear()
            self.events_text.clear()
            
            # Also clear the saved car plates tracking and plate type statistics
            if self.video_thread and hasattr(self.video_thread, 'anpr_processor'):
                self.video_thread.anpr_processor.saved_car_plates.clear()
                # Reset plate type statistics
                for plate_type in self.video_thread.anpr_processor.plate_type_counts:
                    self.video_thread.anpr_processor.plate_type_counts[plate_type] = 0
                logger.info("Cleared saved car plates tracking and plate type statistics")
            
            logger.info("Detection history cleared")
        except Exception as e:
            logger.error(f"Error clearing detections: {e}")
    
    def update_live_feed(self):
        """Update the live feed of detections"""
        try:
            logger.info(f"📺 LIVE FEED: Updating with {len(self.all_detections_history)} total detections in history")
            
            if self.all_detections_history:
                # Show all plate detections (remove confidence filtering)
                visible_detections = [
                    d for d in self.all_detections_history 
                    if d.detection_type == "plate" and d.text not in ["Car", "Plate"]
                ]
                
                logger.info(f"📺 LIVE FEED: Found {len(visible_detections)} visible plate detections")
                
                # Show last 15 visible detections for live feed
                recent_detections = visible_detections[-15:]
                detection_text = "═══ LIVE DETECTION FEED ═══\n\n"
                
                for i, detection in enumerate(recent_detections):
                    timestamp_str = datetime.fromtimestamp(detection.timestamp).strftime('%H:%M:%S')
                    confidence_str = f"{detection.confidence:.3f}" if detection.confidence > 0 else "N/A"
                    
                    # Add plate type information
                    if hasattr(detection, 'plate_type') and detection.plate_type != "unknown":
                        type_emoji = ""
                        if detection.plate_type == "white":
                            type_emoji = "⚪"
                        elif detection.plate_type == "green":
                            type_emoji = "🟢"
                        elif detection.plate_type == "red":
                            type_emoji = "🔴"
                        elif detection.plate_type == "yellow":
                            type_emoji = "🟡"
                        elif detection.plate_type == "blue":
                            type_emoji = "🔵"
                        
                        plate_info = f"{type_emoji}{detection.text} [{detection.plate_type.upper()}]"
                    else:
                        plate_info = f"{detection.text}"
                    
                    # Enhanced format with plate type
                    detection_text += f"{timestamp_str} | {plate_info:<20} | {confidence_str}\n"
                
                self.detections_text.setPlainText(detection_text)
                
                # Auto-scroll to bottom
                scrollbar = self.detections_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
                
                logger.info(f"📺 LIVE FEED: Updated display with {len(recent_detections)} recent detections")
            else:
                # Show waiting message when no detections
                self.detections_text.setPlainText("═══ LIVE DETECTION FEED ═══\n\nWaiting for license plate detections...")
                logger.info("📺 LIVE FEED: No detections in history, showing waiting message")
                
        except Exception as e:
            logger.error(f"Error updating live feed: {e}")
    
    def update_events_feed(self):
        """Update the entry/exit event log."""
        try:
            if self.all_events_history:
                # Show last 20 events
                recent_events = self.all_events_history[-20:]
                log_text = "Timestamp           | Event | Plate Number       | Details\n"
                log_text += "--------------------+-------+--------------------+----------------\n"

                for event in recent_events:
                    ts = datetime.fromtimestamp(event.timestamp).strftime('%Y-%m-%d %H:%M:%S')
                    plate = event.text
                    
                    if event.detection_type == 'plate':
                        event_type = "ENTRY"
                        details = f"Conf: {event.confidence:.2f}, Type: {event.plate_type.upper()}"
                        log_text += f"{ts} | {event_type:<5} | {plate:<18} | {details}\n"
                    elif event.detection_type == 'exit':
                        event_type = "EXIT"
                        details = f"{event.plate_type}" # Duration is stored here
                        log_text += f"{ts} | {event_type:<5} | {plate:<18} | {details}\n"

                self.events_text.setPlainText(log_text)
                scrollbar = self.events_text.verticalScrollBar()
                scrollbar.setValue(scrollbar.maximum())
            else:
                self.events_text.setPlainText("Waiting for vehicle entry/exit events...")
        except Exception as e:
            logger.error(f"Error updating events feed: {e}")
    
    def detect_available_cameras(self):
        """Detect and populate available cameras"""
        try:
            # Show loading message
            self.camera_combo.clear()
            self.camera_combo.addItem("Detecting cameras...")
            self.camera_combo.setEnabled(False)
            QApplication.processEvents()  # Update UI
            
            # Detect cameras
            self.available_cameras = detect_available_cameras()
            
            # Populate camera combo box
            self.camera_combo.clear()
            if self.available_cameras:
                for camera in self.available_cameras:
                    display_text = f"Camera {camera['index']} ({camera['resolution']})"
                    self.camera_combo.addItem(display_text, camera['index'])
                logger.info(f"Found {len(self.available_cameras)} cameras")
            else:
                self.camera_combo.addItem("No cameras found")
                logger.warning("No cameras detected")
            
            # Re-enable combo box
            is_camera = self.source_combo.currentText() == "Camera"
            self.camera_combo.setEnabled(is_camera and len(self.available_cameras) > 0)
            
        except Exception as e:
            logger.error(f"Error detecting cameras: {e}")
            self.camera_combo.clear()
            self.camera_combo.addItem("Error detecting cameras")
    
    def get_camera_config(self):
        """Get current camera configuration from UI"""
        config = {}
        
        # Get resolution
        resolution_text = self.resolution_combo.currentText()
        if "640x480" in resolution_text:
            config['width'] = 640
            config['height'] = 480
        elif "1280x720" in resolution_text:
            config['width'] = 1280
            config['height'] = 720
        elif "1920x1080" in resolution_text:
            config['width'] = 1920
            config['height'] = 1080
        elif "Custom" in resolution_text:
            config['width'] = self.custom_width_spinbox.value()
            config['height'] = self.custom_height_spinbox.value()
        else:
            config['width'] = Config.CAMERA_WIDTH
            config['height'] = Config.CAMERA_HEIGHT
        
        # Get FPS
        config['fps'] = self.camera_fps_spinbox.value()
        
        return config
    
    def get_selected_camera_index(self):
        """Get the currently selected camera index"""
        if self.camera_combo.count() > 0 and self.available_cameras:
            selected_data = self.camera_combo.currentData()
            return selected_data if selected_data is not None else 0
        return 0

    def update_save_detections_status(self, checked):
        """Update the flag for saving detections."""
        Config.SAVE_DETECTIONS = checked
        logger.info(f"Save detections set to: {checked}")

# Utility functions for camera detection
def detect_available_cameras(max_cameras=10):
    """Detect available cameras on the system"""
    available_cameras = []
    
    for i in range(max_cameras):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.isOpened():
                # Try to read a frame to verify camera works
                ret, _ = cap.read()
                if ret:
                    # Get camera info
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    camera_info = {
                        'index': i,
                        'name': f"Camera {i}",
                        'resolution': f"{width}x{height}",
                        'fps': fps if fps > 0 else 30.0
                    }
                    available_cameras.append(camera_info)
                    logger.info(f"Found camera {i}: {width}x{height} @ {fps:.1f}fps")
                cap.release()
            else:
                break  # No more cameras available
        except Exception as e:
            logger.debug(f"Error checking camera {i}: {e}")
            break
    
    return available_cameras

def configure_camera(camera_index, width=None, height=None, fps=None):
    """Configure camera with specified settings"""
    try:
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            return None
        
        # Set camera properties if specified
        if width and height:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            logger.info(f"Set camera {camera_index} resolution to {width}x{height}")
        
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
            logger.info(f"Set camera {camera_index} FPS to {fps}")
        
        # Verify actual settings
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Camera {camera_index} configured: {actual_width}x{actual_height} @ {actual_fps:.1f}fps")
        
        return cap
        
    except Exception as e:
        logger.error(f"Error configuring camera {camera_index}: {e}")
        return None

def main(): 
    """Main application entry point"""
    try:
        app = QApplication(sys.argv)
        
        # Set application properties
        app.setApplicationName("ANPR Live Detection")
        app.setApplicationVersion("1.0")
        
        # Create and show main window
        window = MainWindow()
        window.show()
        
        # Start application
        sys.exit(app.exec_())
        
    except Exception as e:
        logger.error(f"Error starting application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()