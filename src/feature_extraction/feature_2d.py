#!/usr/bin/env python3
"""
2D Feature Extraction from Engineering Drawings.

Extracts dimensions, tolerances, and text from 2D drawings using:
- Image preprocessing (grayscale, binarization, denoising)
- YOLOv7 for object detection
- Tesseract OCR for text extraction
- BERT embeddings for text semantics
"""

import os
import sys
import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


class Node2D:
    """
    Узел 2D-признака с полным набором атрибутов.
    
    Представляет размер, допуск или аннотацию на инженерном чертеже.
    Все поля критически важны для обучения гибридной GNN.
    """
    
    def __init__(self, 
                 node_id: str,
                 feature_type: str,
                 value: float,
                 tolerance: Dict,
                 position_2d: Tuple[float, float],
                 semantic_info: Dict = None,
                 text_embedding: np.ndarray = None,
                 visual_features: np.ndarray = None):
        """
        Инициализация 2D узла.
        
        Args:
            node_id: Уникальный идентификатор узла
            feature_type: Тип признака ('dimension', 'tolerance', 'feature_control_frame', 'datum')
            value: Номинальный размер (например, 10.0 для ⌀10)
            tolerance: Допуск (например, {'type': '±', 'value': 0.1, 'upper_limit': 10.1, 'lower_limit': 9.9})
            position_2d: Позиция на чертеже (x, y)
            semantic_info: Семантическая информация (материал, метод обработки, категория признака)
            text_embedding: Текстовый эмбеддинг (768D из BERT)
            visual_features: Визуальные признаки (256D из ResNet)
        """
        self.id = node_id
        self.type = feature_type  # 'dimension', 'tolerance', 'feature_control_frame', 'datum'
        self.value = value  # номинальный размер
        self.tolerance = tolerance  # допуск с типом и пределами
        self.position_2d = position_2d  # (x, y) на чертеже
        
        # Семантическая информация (критически важна для соответствий!)
        self.semantic_info = semantic_info or {
            'material': None,
            'processing_method': None,
            'feature_category': None,  # hole, slot, pocket, chamfer, fillet и т.д.
            'datum_reference': None,  # ссылка на базу (например, 'A', 'B', 'C')
            'geometric_tolerance_type': None  # для GD&T: flatness, perpendicularity, position и т.д.
        }
        
        # Текстовый эмбеддинг (768D из BERT, как в ViBERTgrid)
        # Кодирует семантическую информацию из текста допусков
        self.text_embedding = text_embedding  # np.array(768,) или None
        
        # Визуальные признаки (256D из ResNet)
        # Дополняет информацию для случаев, когда текст распознан некорректно
        self.visual_features = visual_features  # np.array(256,) или None
        
        # Дополнительные атрибуты
        self.bbox = None  # [x1, y1, x2, y2] bounding box
        self.confidence = 1.0  # уверенность детекции
        self.text_content = None  # исходный текст (например, "⌀10±0.1")
        
        # Связи с другими узлами
        self.parent_dimension = None  # ID родительского размера (для допусков)
        self.related_datums = []  # ID связанных баз
    
    def to_dict(self) -> Dict:
        """Преобразование в словарь для сериализации."""
        return {
            'id': self.id,
            'type': self.type,
            'value': self.value,
            'tolerance': self.tolerance,
            'position_2d': list(self.position_2d) if isinstance(self.position_2d, tuple) else self.position_2d,
            'semantic_info': self.semantic_info,
            'text_embedding': self.text_embedding.tolist() if self.text_embedding is not None else None,
            'visual_features': self.visual_features.tolist() if self.visual_features is not None else None,
            'bbox': self.bbox,
            'confidence': self.confidence,
            'text_content': self.text_content,
            'parent_dimension': self.parent_dimension,
            'related_datums': self.related_datums
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'Node2D':
        """Создание из словаря."""
        node = cls(
            node_id=data['id'],
            feature_type=data['type'],
            value=data['value'],
            tolerance=data['tolerance'],
            position_2d=tuple(data['position_2d']) if isinstance(data['position_2d'], list) else data['position_2d'],
            semantic_info=data.get('semantic_info', {}),
            text_embedding=np.array(data['text_embedding']) if data.get('text_embedding') else None,
            visual_features=np.array(data['visual_features']) if data.get('visual_features') else None
        )
        node.bbox = data.get('bbox')
        node.confidence = data.get('confidence', 1.0)
        node.text_content = data.get('text_content')
        node.parent_dimension = data.get('parent_dimension')
        node.related_datums = data.get('related_datums', [])
        return node
    
    def get_tolerance_status(self, measured_value: float) -> str:
        """
        Определить статус соответствия по измеренному значению.
        
        Args:
            measured_value: Измеренное значение из 3D облака точек
            
        Returns:
            'IN_TOLERANCE', 'MARGINAL', или 'OUT_OF_TOLERANCE'
        """
        if not self.tolerance:
            return 'UNKNOWN'
        
        upper = self.tolerance.get('upper_limit', self.value)
        lower = self.tolerance.get('lower_limit', self.value)
        
        if lower <= measured_value <= upper:
            return 'IN_TOLERANCE'
        elif abs(measured_value - upper) <= 0.1 * abs(upper - lower) or \
             abs(measured_value - lower) <= 0.1 * abs(upper - lower):
            return 'MARGINAL'
        else:
            return 'OUT_OF_TOLERANCE'
    
    def get_tolerance_range(self) -> float:
        """Получить диапазон допуска (верхний - нижний предел)."""
        if not self.tolerance:
            return 0.0
        upper = self.tolerance.get('upper_limit', self.value)
        lower = self.tolerance.get('lower_limit', self.value)
        return abs(upper - lower)


class ImagePreprocessor:
    """
    Preprocess engineering drawing images.
    
    Implements:
    - Grayscale conversion
    - Binarization (Otsu)
    - Denoising (morphological operations)
    - Perspective correction
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'blur_kernel': 5,
            'morph_kernel': 3,
            'morph_iterations': 1,
            'min_area': 50
        }
    
    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for dimension detection."""
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (self.config['blur_kernel'], 
                                          self.config['blur_kernel']), 0)
        
        # Otsu binarization
        _, binary = cv2.threshold(blurred, 0, 255, 
                                 cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = np.ones((self.config['morph_kernel'], 
                         self.config['morph_kernel']), np.uint8)
        denoised = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, 
                                   iterations=self.config['morph_iterations'])
        
        # Remove small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(denoised)
        cleaned = np.zeros_like(denoised)
        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.config['min_area']:
                cleaned[labels == i] = 255
        
        return cleaned


class DimensionDetector:
    """
    Detect dimensions and tolerances using YOLOv7.

    Classes:
    - dimension (linear, angular)
    - tolerance (±, upper/lower limits)
    - feature_control_frame
    - datum
    - text
    """

    def __init__(self, config: Dict = None):
        self.config = config or {
            'model_path': '/home/spectr/itmo/recg_drawing/yolov7/weights/0324_dim_and_tol_best.pt',
            'confidence_threshold': 0.5,
            'classes': ['dimension', 'tolerance', 'fcf', 'datum', 'text']
        }

        self.model = None
        self.device = None
        if self.config.get('model_path') and os.path.exists(self.config['model_path']):
            self._load_model()

    def _load_model(self):
        """Load YOLOv7 model from recg_drawing."""
        try:
            import torch
            
            # Patch torch.load for YOLOv7 compatibility BEFORE any torch operations
            if not hasattr(torch, '_original_load'):
                torch._original_load = torch.load
                def _patched_torch_load(*args, **kwargs):
                    if 'weights_only' not in kwargs:
                        kwargs['weights_only'] = False
                    return torch._original_load(*args, **kwargs)
                torch.load = _patched_torch_load
            
            # Change to yolov7 directory for proper imports
            yolov7_path = '/home/spectr/itmo/recg_drawing/yolov7'
            original_cwd = os.getcwd()
            os.chdir(yolov7_path)

            if yolov7_path not in sys.path:
                sys.path.insert(0, yolov7_path)
            
            # Load model
            from models.experimental import attempt_load
            from utils.torch_utils import select_device
            
            model_path = self.config['model_path']
            self.device = select_device('')
            self.model = attempt_load(model_path, map_location=self.device)
            self.model.eval()
            
            # Restore original directory
            os.chdir(original_cwd)

            print(f"    Loaded YOLOv7 from {self.config['model_path']}")
            print(f"    Classes: {self.config['classes']}")
        except Exception as e:
            print(f"  ⚠ Failed to load YOLOv7 model: {e}. Using synthetic detection.")
            import traceback
            traceback.print_exc()

    def detect(self, image: np.ndarray) -> List[Dict]:
        """Detect dimensions and tolerances."""
        if self.model is None:
            return self._synthetic_detect(image)

        # Run YOLOv7 inference using recg_drawing approach
        return self._yolov7_detect(image)

    def _yolov7_detect(self, image: np.ndarray) -> List[Dict]:
        """YOLOv7 detection using recg_drawing approach."""
        try:
            import torch
            from utils.general import check_img_size, non_max_suppression, scale_coords
            from utils.datasets import letterbox
            
            # Get model parameters
            stride = int(self.model.stride.max())
            img_size = check_img_size(self.config.get('img_size', 640), stride)
            
            # Preprocess image - handle img_size as int or tuple
            if isinstance(img_size, int):
                img_size = (img_size, img_size)
            img = letterbox(image, img_size)[0]
            img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, HWC to CHW
            img = np.ascontiguousarray(img)
            img = torch.from_numpy(img).to(self.device)
            img = img.float() / 255.0  # Normalize
            if img.ndimension() == 3:
                img = img.unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                pred = self.model(img, augment=False)[0]
            
            # Apply NMS
            pred = non_max_suppression(pred, self.config['confidence_threshold'], 0.45)
            
            detections = []
            h, w = image.shape[:2]
            
            for det in pred:
                if det is not None and len(det):
                    # Rescale boxes from img_size to image size
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], image.shape).round()
                    
                    for *xyxy, conf, cls in det:
                        # Get class name - handle torch tensor
                        cls_int = int(cls.item()) if hasattr(cls, 'item') else int(cls)
                        if isinstance(self.config['classes'], list) and cls_int < len(self.config['classes']):
                            class_name = self.config['classes'][cls_int]
                        else:
                            class_name = str(cls_int)
                        
                        det_dict = {
                            'class': class_name,
                            'confidence': float(conf),
                            'bbox': [float(x) for x in xyxy]
                        }
                        detections.append(det_dict)
            
            return detections if detections else self._synthetic_detect(image)
            
        except Exception as e:
            print(f"  ⚠ YOLOv7 detection failed: {e}. Using synthetic detection.")
            import traceback
            traceback.print_exc()
            return self._synthetic_detect(image)

    def _load_model_with_weights_only_false(self, model_path, device):
        """Load YOLOv7 model with weights_only=False for PyTorch 2.6+."""
        import torch
        from models.yolo import Model
        from models.experimental import attempt_load
        
        # Try loading with weights_only=False for PyTorch 2.6+
        try:
            # For newer PyTorch versions, we need to load checkpoint manually
            ckpt = torch.load(model_path, map_location=device, weights_only=False)
            model = ckpt['model'].float()
            model.to(device).eval()
            return model
        except Exception as e:
            print(f"  ⚠ Failed to load with weights_only=False: {e}")
            return None

    def _letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        """Resize and pad image."""
        shape = img.shape[:2]  # current shape [height, width]
        
        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        
        # Compute padding
        new_unpad = int(shape[1] * r), int(shape[0] * r)
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
        
        # Divide padding into 2 sides
        dw /= 2
        dh /= 2
        
        # Resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
        
        return img, (dw, dh)

    def _synthetic_detect(self, image: np.ndarray) -> List[Dict]:
        """Synthetic detection for testing."""
        h, w = image.shape[:2]

        return [
            {
                'class': 'dimension',
                'confidence': 0.9,
                'bbox': [w * 0.4, h * 0.4, w * 0.6, h * 0.5]
            },
            {
                'class': 'tolerance',
                'confidence': 0.85,
                'bbox': [w * 0.45, h * 0.42, w * 0.55, h * 0.48]
            }
        ]


class OCRExtractor:
    """
    Extract text from detected bounding boxes using Tesseract OCR.
    """

    def __init__(self, config: Dict = None):
        self.config = config or {}
        # Set defaults
        self.lang = self.config.get('lang', 'eng')
        self.psm = self.config.get('psm', 7)
        self.oem = self.config.get('oem', 3)
        self.preprocess = self.config.get('preprocess', True)
        self.scale = self.config.get('scale', 2.0)

    def extract(self, image: np.ndarray, bbox: List[float]) -> str:
        """Extract text from bounding box with preprocessing."""
        try:
            import pytesseract
            import cv2

            x1, y1, x2, y2 = map(int, bbox)
            roi = image[y1:y2, x1:x2]

            # Skip if ROI is too small
            if roi.size == 0 or roi.shape[0] < 10 or roi.shape[1] < 10:
                return ""

            # Preprocess image for better OCR
            if self.preprocess:
                roi = self._preprocess_roi(roi)

            # Configure Tesseract
            config_str = f"--oem {self.oem} --psm {self.psm}"

            # Extract text
            text = pytesseract.image_to_string(roi,
                                               lang=self.lang,
                                               config=config_str)

            # Clean up text
            text = self._clean_text(text)

            return text.strip()
        except ImportError:
            print("  ⚠ pytesseract not installed. Install with: pip install pytesseract")
            return ""
        except Exception as e:
            print(f"  ⚠ OCR error: {e}")
            return ""

    def _preprocess_roi(self, roi: np.ndarray) -> np.ndarray:
        """Preprocess ROI for better OCR accuracy."""
        import cv2

        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi.copy()

        # Add white border to prevent edge clipping
        border_size = max(10, int(min(gray.shape[:2]) * 0.1))
        gray = cv2.copyMakeBorder(gray, border_size, border_size, border_size, border_size,
                                  cv2.BORDER_CONSTANT, value=255)

        # Scale up significantly for better recognition
        scale = self.config.get('scale', 3.0)
        if scale != 1.0:
            gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

        # Apply Gaussian blur to reduce noise
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # Try Otsu's thresholding (better for uniform backgrounds)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Morphological operations to clean up
        kernel = np.ones((2, 2), np.uint8)
        
        # Close small gaps
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel, iterations=1)
        
        # Remove small noise
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Invert if text is white on black (Tesseract expects black text on white)
        # Check if most pixels are dark (text is light)
        mean_pixel = np.mean(thresh)
        if mean_pixel < 127:
            thresh = cv2.bitwise_not(thresh)

        # Additional cleanup: remove very small objects
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(thresh)
        min_area = 5  # Minimum area for a valid component
        clean_thresh = np.zeros_like(thresh)
        for i in range(1, num_labels):  # Skip background (label 0)
            if stats[i, cv2.CC_STAT_AREA] >= min_area:
                clean_thresh[labels == i] = 255

        return clean_thresh

    def _clean_text(self, text: str) -> str:
        """Clean OCR text output."""
        import re

        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove common OCR artifacts
        text = text.replace('?', '').replace('|', '').replace('!', '')

        # Replace common misrecognitions
        replacements = {
            'o': '0',  # letter o to zero (in numeric context)
            'O': '0',  # letter O to zero
            'l': '1',  # letter l to one (in numeric context)
            'I': '1',  # letter I to one
            '±': '±',  # normalize plus-minus
            '+/-': '±',  # alternative plus-minus
        }

        for old, new in replacements.items():
            text = text.replace(old, new)

        return text.strip()

    def parse_tolerance(self, text: str) -> Dict:
        """
        Parse tolerance from OCR text.

        Examples:
        - "10.0 ±0.1" -> {'nominal': 10.0, 'type': 'bilateral', 'value': 0.1, 'upper_limit': 10.1, 'lower_limit': 9.9}
        - "10.0 +0.2/-0.1" -> {'nominal': 10.0, 'type': 'limit', 'upper': 0.2, 'lower': 0.1, 'upper_limit': 10.2, 'lower_limit': 9.9}
        - "⌀10 H7" -> {'nominal': 10.0, 'type': 'fit', 'fit_class': 'H7'}
        """
        import re

        if not text:
            return {'nominal': 0, 'type': 'unknown', 'value': 0, 'upper_limit': 0, 'lower_limit': 0}

        # Pattern for ± tolerance: "10.0 ±0.1" or "10±0.1"
        match = re.search(r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)', text)
        if match:
            nominal = float(match.group(1))
            value = float(match.group(2))
            return {
                'nominal': nominal,
                'type': 'bilateral',
                'value': value,
                'upper_limit': nominal + value,
                'lower_limit': nominal - value,
                'display': f'±{value}'
            }

        # Pattern for limit dimensioning: "10.0 +0.2/-0.1" or "10 +0.2/-0.1"
        match = re.search(r'(\d+\.?\d*)\s*\+\s*(\d+\.?\d*)\s*/\s*-\s*(\d+\.?\d*)', text)
        if match:
            nominal = float(match.group(1))
            upper = float(match.group(2))
            lower = float(match.group(3))
            return {
                'nominal': nominal,
                'type': 'limit',
                'upper': upper,
                'lower': lower,
                'upper_limit': nominal + upper,
                'lower_limit': nominal - lower,
                'display': f'+{upper}/-{lower}'
            }

        # Pattern for diameter: "⌀10" or "10" or "DIA 10"
        match = re.search(r'[⌀DIA\d]*\s*(\d+\.?\d*)', text, re.IGNORECASE)
        if match:
            value = float(match.group(1))
            return {
                'nominal': value,
                'type': 'diameter' if '⌀' in text or 'DIA' in text.upper() else 'linear',
                'value': 0,
                'upper_limit': value,
                'lower_limit': value,
                'display': str(value)
            }

        # Pattern for fit tolerance: "10 H7" or "10h7"
        match = re.search(r'(\d+\.?\d*)\s*([Hh][0-9]|[Gg][0-9])', text)
        if match:
            nominal = float(match.group(1))
            fit_class = match.group(2)
            return {
                'nominal': nominal,
                'type': 'fit',
                'fit_class': fit_class,
                'value': 0,
                'upper_limit': nominal,
                'lower_limit': nominal,
                'display': f'{nominal} {fit_class}'
            }

        # Fallback: try to extract any number
        match = re.search(r'(\d+\.?\d*)', text)
        if match:
            value = float(match.group(1))
            return {
                'nominal': value,
                'type': 'unknown',
                'value': 0,
                'upper_limit': value,
                'lower_limit': value,
                'display': str(value)
            }

        return {'nominal': 0, 'type': 'unknown', 'value': 0, 'upper_limit': 0, 'lower_limit': 0}

    def extract_value_from_bbox(self, image: np.ndarray, bbox: List[float]) -> Tuple[float, Dict, str]:
        """
        Extract value, tolerance, and raw text from bounding box.

        Args:
            image: Source image
            bbox: [x1, y1, x2, y2] bounding box

        Returns:
            Tuple of (nominal_value, tolerance_dict, raw_text)
        """
        # Extract text
        text = self.extract(image, bbox)

        # Parse tolerance
        tolerance = self.parse_tolerance(text)

        # Get nominal value
        nominal = tolerance.get('nominal', 0)

        return nominal, tolerance, text

class TextEmbeddingExtractor:
    """
    Extract BERT embeddings from text content.
    
    Uses bert-base-uncased from ViBERTgrid-PyTorch to encode text into 768D vector.
    """
    
    def __init__(self, model_name: str = 'bert-base-uncased'):
        self.model = None
        self.tokenizer = None
        self.model_name = model_name
        self.device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load BERT model and tokenizer."""
        try:
            import torch
            from transformers import BertModel, BertTokenizer
            
            print(f"  [BERT] Loading {self.model_name}...")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
            self.model = BertModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            print(f"  [BERT] Loaded on {self.device}")
        except ImportError:
            print("  ⚠ transformers not installed. Install with: pip install transformers")
        except Exception as e:
            print(f"  ⚠ Failed to load BERT: {e}")
    
    def extract_embedding(self, text: str) -> np.ndarray:
        """
        Extract 768D BERT embedding from text.
        
        Args:
            text: Input text (e.g., "207.260/-7.66")
            
        Returns:
            768D numpy array (CLS token embedding)
        """
        if self.model is None:
            self.load_model()
        
        if self.model is None:
            # Return dummy embedding if model failed to load
            return np.zeros(768, dtype=np.float32)
        
        import torch
        
        # Tokenize
        inputs = self.tokenizer(
            text, 
            return_tensors='pt', 
            truncation=True, 
            padding=True, 
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get embeddings
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use [CLS] token embedding (768D)
        cls_embedding = outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy()
        
        return cls_embedding


class VisualFeatureExtractor:
    """
    Extract visual features from detected regions using ResNet.
    
    Uses pre-trained ResNet-18 to encode visual appearance into 256D vector.
    """
    
    def __init__(self, model_name: str = 'resnet18'):
        self.feature_extractor = None
        self.projection = None
        self.model_name = model_name
        self.transform = None
        self.device = 'cuda' if __import__('torch').cuda.is_available() else 'cpu'
    
    def load_model(self):
        """Load ResNet model."""
        try:
            import torch
            import torch.nn as nn
            from torchvision import models, transforms
            
            print(f"  [ResNet] Loading {self.model_name}...")
            
            # Load pre-trained ResNet
            if self.model_name == 'resnet18':
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            elif self.model_name == 'resnet34':
                resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
            elif self.model_name == 'resnet50':
                resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            else:
                resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            
            # Remove final classification layer (fc)
            # Use features before fc layer (512D for ResNet-18/34, 2048D for ResNet-50)
            self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
            self.feature_extractor.to(self.device)
            self.feature_extractor.eval()
            
            # Add projection layer to reduce to 256D
            self.projection = nn.Linear(512, 256)
            self.projection.to(self.device)
            self.projection.eval()
            
            # Define preprocessing
            self.transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                   std=[0.229, 0.224, 0.225])
            ])
            
            print(f"  [ResNet] Loaded on {self.device}")
        except ImportError:
            print("  ⚠ torchvision not installed. Install with: pip install torchvision")
        except Exception as e:
            print(f"  ⚠ Failed to load ResNet: {e}")
    
    def extract_features(self, image: np.ndarray, bbox: List[float] = None) -> np.ndarray:
        """
        Extract 256D visual features from ROI.
        
        Args:
            image: Input image
            bbox: [x1, y1, x2, y2] bounding box (optional, uses full image if None)
            
        Returns:
            256D numpy array
        """
        if self.feature_extractor is None:
            self.load_model()
        
        if self.feature_extractor is None:
            return np.zeros(256, dtype=np.float32)
        
        import torch
        
        # Extract ROI or use full image
        if bbox is not None:
            x1, y1, x2, y2 = map(int, bbox)
            # Ensure bbox is within image bounds
            h, w = image.shape[:2]
            x1, x2 = max(0, x1), min(w, x2)
            y1, y2 = max(0, y1), min(h, y2)
            
            if x2 <= x1 or y2 <= y1:
                return np.zeros(256, dtype=np.float32)
            
            roi = image[y1:y2, x1:x2]
        else:
            roi = image
        
        if roi.size == 0 or roi.shape[0] < 5 or roi.shape[1] < 5:
            return np.zeros(256, dtype=np.float32)
        
        # Convert BGR to RGB if needed
        if len(roi.shape) == 3 and roi.shape[2] == 3:
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
        
        # Preprocess
        roi_tensor = self.transform(roi)
        roi_tensor = roi_tensor.unsqueeze(0).to(self.device)
        
        # Extract features
        with torch.no_grad():
            features = self.feature_extractor(roi_tensor)
            features = features.squeeze()  # 512D for ResNet-18
        
        # Convert to numpy and ensure it's 256D
        features_np = features.cpu().numpy()
        
        # Ensure correct shape: use first 256 dimensions
        if len(features_np.shape) == 0:
            # Scalar case
            return np.zeros(256, dtype=np.float32)
        elif len(features_np.shape) == 1:
            if len(features_np) >= 256:
                return features_np[:256].astype(np.float32)
            else:
                # Pad with zeros
                result = np.zeros(256, dtype=np.float32)
                result[:len(features_np)] = features_np
                return result
        else:
            # Multi-dimensional: flatten and take first 256
            flat = features_np.flatten()
            return flat[:256].astype(np.float32) if len(flat) >= 256 else np.zeros(256, dtype=np.float32)


class FeatureExtractor2D:
    """
    Complete 2D feature extraction pipeline.
    
    Combines:
    - Image preprocessing
    - YOLOv7 detection
    - OCR extraction
    - BERT text embeddings
    - ResNet visual features
    - Node creation with full attribute set
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {}
        
        self.preprocessor = ImagePreprocessor(config)
        self.detector = DimensionDetector(config)
        self.ocr_extractor = OCRExtractor(config)
        
        # Optional: BERT embeddings and ResNet features
        self.text_embedder = None
        self.visual_extractor = None
        
        if self.config.get('use_bert', False):
            self.text_embedder = TextEmbeddingExtractor()
        
        if self.config.get('use_resnet', False):
            self.visual_extractor = VisualFeatureExtractor()
        
        self.node_counter = 0
    
    def extract_features(self, drawing_path: str,
                        image: np.ndarray = None) -> List[Node2D]:
        """
        Extract 2D features from drawing.
        Uses SVG direct parsing if available, otherwise YOLOv7+OCR (disabled temporarily).
        """
        drawing_path = Path(drawing_path)
        
        # Try SVG direct parsing first
        svg_path = None
        if drawing_path.suffix == '.svg':
            svg_path = drawing_path
        elif drawing_path.suffix == '.png':
            svg_path = drawing_path.with_suffix('.svg')
        
        if svg_path and svg_path.exists():
            print(f"  [SVG] Parsing {svg_path.name} directly for dimensions")
            return self._extract_from_svg(svg_path)
        
        # Fallback to image (YOLOv7 disabled - needs retraining)
        print(f"  [YOLOv7] Disabled - needs retraining on this dataset")
        return []

    def _extract_from_svg(self, svg_path: Path) -> List[Node2D]:
        """Extract dimensions directly from SVG text elements."""
        import re
        import xml.etree.ElementTree as ET
        import cv2
        
        # Parse SVG
        tree = ET.parse(svg_path)
        root = tree.getroot()
        
        # Define namespace (SVG usually has xmlns)
        ns = {'svg': 'http://www.w3.org/2000/svg'}
        
        # Find all text elements
        text_elements = root.findall('.//svg:text', ns)
        if not text_elements:
            # Try without namespace
            text_elements = root.findall('.//text')
            if not text_elements:
                # Try with full path
                text_elements = list(root.iter('{http://www.w3.org/2000/svg}text'))
        
        # Load corresponding PNG image for visual features
        png_path = svg_path.with_suffix('.png')
        image = None
        if png_path.exists():
            image = cv2.imread(str(png_path))
        
        nodes = []
        
        for text_elem in text_elements:
            # Get text content
            text = text_elem.text or ''
            text = text.strip()
            
            if not text:
                continue
            
            # 0. Skip GD&T symbols (⌖, ⊥, ⊕, ◎, ∠, ∥, ○, ⌒, ⌓, ⌭, ⌮, ⌯, ⌭)
            # These are geometric tolerance symbols, not dimensions
            gd_t_symbols = {'⌖', '⊥', '⊕', '◎', '∠', '∥', '○', '⌒', '⌓', '⌭', '⌮', '⌯', '⌀', '↗', '↘', '↙', '↖'}
            if text in gd_t_symbols or (len(text) == 1 and any(ord(c) > 0x2300 for c in text)):
                # Single GD&T symbol - skip it
                continue
            
            # Get position from x/y attributes
            x = float(text_elem.get('x', '0'))
            y = float(text_elem.get('y', '0'))
            
            # Get rotation from transform
            transform = text_elem.get('transform', '')
            rotation = 0
            transform_match = re.search(r'rotate\(([-\d.]+)', transform)
            if transform_match:
                rotation = float(transform_match.group(1))
            
            # 1. Check for datum references (single uppercase letters A-Z, or sequences like "A B C")
            # Datums are typically uppercase letters, possibly separated by spaces
            # First check if it's a single letter
            if re.match(r'^[A-Z]$', text):
                # This is a single datum reference
                self.node_counter += 1
                node = Node2D(
                    node_id=f"node2d_{self.node_counter:03d}",
                    feature_type='datum',
                    value=0,
                    tolerance={
                        'type': 'datum_reference',
                        'datum_letter': text,
                        'nominal': 0,
                        'value': 0,
                        'display': text
                    },
                    position_2d=(x, y),
                    semantic_info={
                        'feature_category': 'datum',
                        'datum_letter': text,
                        'extraction_method': 'svg_parsing',
                        'rotation': rotation
                    },
                    text_embedding=text_embedding,
                    visual_features=visual_features
                )
                node.text_content = text
                node.confidence = 1.0
                node.bbox = [x - 15, y - 15, x + 15, y + 15]
                
                nodes.append(node)
                print(f"  [SVG] Datum: '{text}' at ({x:.1f}, {y:.1f})")
                continue
            
            # Check for compound datum reference (e.g., "A B C" or "A-B-C")
            datum_match = re.match(r'^([A-Z](?:\s+[A-Z]|\s*-\s*[A-Z])*)$', text)
            if datum_match:
                # Split into individual datums and create nodes for each
                datum_letters = re.findall(r'[A-Z]', text)
                if len(datum_letters) > 0:
                    for i, letter in enumerate(datum_letters):
                        self.node_counter += 1
                        node = Node2D(
                            node_id=f"node2d_{self.node_counter:03d}",
                            feature_type='datum',
                            value=0,
                            tolerance={
                                'type': 'datum_reference',
                                'datum_letter': letter,
                                'nominal': 0,
                                'value': 0,
                                'display': letter
                            },
                            position_2d=(x + i * 20, y),  # Offset positions slightly
                            semantic_info={
                                'feature_category': 'datum',
                                'datum_letter': letter,
                                'extraction_method': 'svg_parsing',
                                'rotation': rotation,
                                'compound_datum': len(datum_letters) > 1,
                                'datum_sequence': datum_letters
                            },
                            text_embedding=text_embedding,
                            visual_features=visual_features
                        )
                        node.text_content = letter
                        node.confidence = 1.0
                        node.bbox = [x + i * 20 - 15, y - 15, x + i * 20 + 15, y + 15]
                        
                        nodes.append(node)
                    print(f"  [SVG] Compound datum: '{text}' → {datum_letters} at ({x:.1f}, {y:.1f})")
                    continue
            
            # 2. Check for GD&T symbols with numbers (e.g., "⌀0.5", "⌖ 0.5")
            # Remove ⌀ but treat as diameter dimension
            is_diameter = '⌀' in text
            cleaned_text = text.replace('⌀', '').strip()
            
            # Skip text that contains only GD&T symbols
            if re.match(r'^[⌖⊥⊕◎∠∥○⌒⌓⌭⌮⌯↗↘↙↖\s]*$', text):
                # Only GD&T symbols, no numbers - skip
                continue
            
            # 3. Parse dimension values and tolerances
            nominal, tolerance, feature_hint = self._parse_dimension_text(text)
            
            # Determine feature category based on text pattern
            if is_diameter:
                feature_category = 'hole'  # Diameter typically indicates hole or cylinder
            elif feature_hint:
                feature_category = feature_hint
            elif '±' in text or '/' in text:
                feature_category = 'dimension_with_tolerance'
            else:
                feature_category = 'dimension'
            
            # Extract text embedding
            text_embedding = None
            if self.text_embedder:
                text_embedding = self.text_embedder.extract_embedding(text)
            
            # Extract visual features
            visual_features = None
            if self.visual_extractor and image is not None:
                h, w = image.shape[:2]
                
                # Get SVG viewBox for coordinate mapping
                svg_viewbox = root.get('viewBox')
                
                if svg_viewbox:
                    # viewBox can use spaces or commas as separators
                    # Replace commas with spaces, then split
                    parts = svg_viewbox.replace(',', ' ').split()
                    if len(parts) == 4:
                        vb_x, vb_y, vb_w, vb_h = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
                        scale_x = w / vb_w if vb_w > 0 else 1
                        scale_y = h / vb_h if vb_h > 0 else 1
                        pixel_x = (x - vb_x) * scale_x
                        pixel_y = (y - vb_y) * scale_y
                    else:
                        pixel_x, pixel_y = x, y
                else:
                    pixel_x, pixel_y = x, y
                
                # Use a fixed ROI size (80x40 pixels) centered on the text position
                roi_size_x, roi_size_y = 80, 40
                x1 = int(pixel_x - roi_size_x/2)
                y1 = int(pixel_y - roi_size_y/2)
                x2 = int(pixel_x + roi_size_x/2)
                y2 = int(pixel_y + roi_size_y/2)
                
                # Clamp to image bounds
                x1, x2 = max(0, x1), min(w, x2)
                y1, y2 = max(0, y1), min(h, y2)
                
                # Skip if ROI is too small
                if x2 - x1 < 10 or y2 - y1 < 10:
                    visual_features = None
                else:
                    bbox = [x1, y1, x2, y2]
                    visual_features = self.visual_extractor.extract_features(image, bbox)
            
            # Create Node2D with full semantic info
            self.node_counter += 1
            node = Node2D(
                node_id=f"node2d_{self.node_counter:03d}",
                feature_type=feature_category,
                value=nominal,
                tolerance=tolerance,
                position_2d=(x, y),
                semantic_info={
                    'feature_category': feature_category,
                    'extraction_method': 'svg_parsing',
                    'rotation': rotation,
                    'is_diameter': is_diameter,
                    'has_tolerance': tolerance.get('value', 0) > 0,
                    'raw_text': text
                },
                text_embedding=text_embedding,
                visual_features=visual_features
            )
            node.text_content = text
            node.confidence = 1.0  # SVG parsing is 100% accurate
            node.bbox = [x - 30, y - 15, x + 30, y + 15]  # Approximate bbox

            nodes.append(node)
            emb_info = ""
            if text_embedding is not None:
                emb_info += f", BERT={text_embedding.shape}"
            if visual_features is not None:
                emb_info += f", ResNet={visual_features.shape}"
            print(f"  [SVG] Parsed: '{text}' at ({x:.1f}, {y:.1f}), value={nominal}, tol={tolerance.get('value', 0):.3f}{emb_info}")
        
        return nodes
    
    def _parse_dimension_text(self, text: str) -> tuple:
        """
        Parse dimension text to extract nominal value and tolerance.
        
        Returns:
            (nominal, tolerance_dict, feature_hint)
        """
        import re

        # Remove ⌀ symbol but remember it's a diameter
        cleaned = text.replace('⌀', '').replace('°', '').strip()
        
        # Pattern 1: "NOMINAL±TOLERANCE" (e.g., "207.26±5.63")
        pattern_symmetric = r'(\d+\.?\d*)\s*±\s*(\d+\.?\d*)'
        match_sym = re.match(pattern_symmetric, cleaned)
        
        if match_sym:
            nominal = float(match_sym.group(1))
            tol_value = float(match_sym.group(2))
            
            return nominal, {
                'type': 'symmetric',
                'nominal': nominal,
                'value': tol_value,
                'upper_limit': nominal + tol_value,
                'lower_limit': nominal - tol_value,
                'display': text
            }, None
        
        # Pattern 2: "NOMINAL/LOWER" or "NOMINAL+UPPER/LOWER" (e.g., "207.260/-7.66", "205.23+5.20/0")
        pattern_asymmetric = r'(\d+\.?\d*)\s*([+-]\s*\d+\.?\d*)?\s*/\s*([+-]?\s*\d+\.?\d*)'
        match_asym = re.match(pattern_asymmetric, cleaned)
        
        if match_asym:
            nominal = float(match_asym.group(1))
            upper_str = match_asym.group(2)
            lower_str = match_asym.group(3)
            
            # If upper_str is None, the format is "NOMINAL/LOWER" meaning upper=0
            if upper_str:
                upper_str = upper_str.replace(' ', '')
                upper_val = float(upper_str)
            else:
                upper_val = 0.0
            
            lower_str = lower_str.replace(' ', '')
            lower_val = float(lower_str)
            
            return nominal, {
                'type': 'asymmetric',
                'nominal': nominal,
                'value': abs(upper_val - lower_val) / 2,
                'upper_limit': nominal + upper_val,
                'lower_limit': nominal + lower_val,
                'upper_deviation': upper_val,
                'lower_deviation': lower_val,
                'display': text
            }, None
        
        # Pattern 3: Just a number (e.g., "0.5" for ⌀0.5)
        pattern_number = r'(\d+\.?\d*)'
        match_num = re.match(pattern_number, cleaned)
        
        if match_num:
            nominal = float(match_num.group(1))
            
            return nominal, {
                'type': 'nominal_only',
                'nominal': nominal,
                'value': 0,
                'upper_limit': nominal,
                'lower_limit': nominal,
                'display': text
            }, None
        
        # No number found
        return 0, {
            'type': 'unknown',
            'nominal': 0,
            'value': 0,
            'upper_limit': 0,
            'lower_limit': 0,
            'display': text
        }, None

    def _extract_from_json(self, json_path: Path) -> List[Node2D]:
        """Extract features from JSON annotation."""
        with open(json_path, 'r') as f:
            drawing = json.load(f)
        
        nodes = []
        
        # Extract from each view
        for view_name, view_data in drawing.items():
            if not isinstance(view_data, dict):
                continue
            
            dimensions = view_data.get('dimensions', [])
            for dim in dimensions:
                node = self._create_node_from_dimension(dim)
                nodes.append(node)
        
        return nodes
    
    def _extract_from_image(self, image: np.ndarray) -> List[Node2D]:
        """Extract features from image with full attribute extraction including OCR."""
        # Preprocess
        preprocessed = self.preprocessor.preprocess(image)

        # Detect dimensions using YOLOv7
        detections = self.detector.detect(image)

        # Extract attributes and create nodes
        nodes = []
        for i, det in enumerate(detections):
            # Extract text using OCR
            text = self.ocr_extractor.extract(image, det['bbox'])

            # Parse tolerance and extract value
            nominal, tolerance, raw_text = self.ocr_extractor.extract_value_from_bbox(
                image, det['bbox']
            )
            
            # Filter out datum symbols (single uppercase letters)
            if raw_text and len(raw_text.strip()) == 1 and raw_text.strip().isalpha():
                # This is likely a datum symbol, not a dimension
                continue

            # Extract BERT embedding (if enabled)
            text_embedding = None
            if self.text_embedder:
                text_embedding = self.text_embedder.extract_embedding(text)

            # Extract visual features (if enabled)
            visual_features = None
            if self.visual_extractor:
                visual_features = self.visual_extractor.extract_features(image, det['bbox'])

            # Infer semantic info
            semantic_info = self._infer_semantic_info(det['class'], text, tolerance)

            # Create node with full attribute set
            self.node_counter += 1
            node = Node2D(
                node_id=f"node2d_{self.node_counter:03d}",
                feature_type=det['class'],
                value=nominal,  # Use OCR-extracted value
                tolerance=tolerance,  # Use OCR-extracted tolerance
                position_2d=(
                    (det['bbox'][0] + det['bbox'][2]) / 2,
                    (det['bbox'][1] + det['bbox'][3]) / 2
                ),
                semantic_info=semantic_info,
                text_embedding=text_embedding,
                visual_features=visual_features
            )

            # Set additional attributes
            node.bbox = det['bbox']
            node.confidence = det['confidence']
            node.text_content = raw_text  # Store raw OCR text

            nodes.append(node)

        return nodes
    def _create_node_from_dimension(self, dim: Dict) -> Node2D:
        """Create Node2D from dimension dictionary with full attributes."""
        self.node_counter += 1
        
        # Extract or create semantic info
        semantic_info = dim.get('semantic_info', {})
        if not semantic_info:
            semantic_info = {
                'material': None,
                'processing_method': None,
                'feature_category': dim.get('feature_type', 'unknown'),
                'datum_reference': dim.get('datum_reference'),
                'geometric_tolerance_type': None
            }
        
        node = Node2D(
            node_id=dim.get('id', f"node2d_{self.node_counter:03d}"),
            feature_type='dimension',
            value=dim.get('value', 0),
            tolerance=dim.get('tolerance', {}),
            position_2d=tuple(dim.get('position_2d', [0, 0])),
            semantic_info=semantic_info,
            text_embedding=None,  # Would be computed if BERT enabled
            visual_features=None  # Would be computed if ResNet enabled
        )
        
        node.bbox = dim.get('bounding_box')
        node.confidence = 1.0
        node.text_content = dim.get('text', '')
        
        return node
    
    def _infer_semantic_info(self, det_class: str, text: str, 
                            tolerance: Dict) -> Dict:
        """Infer semantic information from detection."""
        # Infer feature category from text patterns
        feature_category = 'unknown'
        
        if '⌀' in text or 'dia' in text.lower() or 'diameter' in text.lower():
            feature_category = 'hole'
        elif 'R' in text or 'radius' in text.lower():
            feature_category = 'fillet'
        elif '□' in text or 'square' in text.lower():
            feature_category = 'slot'
        elif '⌖' in text or 'position' in text.lower():
            feature_category = 'pattern'
        elif '⟂' in text or 'perpendicular' in text.lower():
            feature_category = 'surface'
        elif '⏊' in text or 'flatness' in text.lower():
            feature_category = 'plane'
        
        # Infer datum reference
        datum_reference = None
        if det_class == 'datum':
            datum_reference = text.strip() if text else None
        
        # Infer geometric tolerance type
        geo_tol_type = None
        if det_class == 'fcf':
            geo_symbols = {
                '⏊': 'flatness',
                '○': 'circularity',
                '⌭': 'cylindricity',
                '⌓': 'profile_of_surface',
                '∠': 'angularity',
                '⟂': 'perpendicularity',
                '∥': 'parallelism',
                '⌖': 'position',
                '◎': 'concentricity',
                '↗': 'circular_runout'
            }
            for symbol, tol_type in geo_symbols.items():
                if symbol in text:
                    geo_tol_type = tol_type
                    break
        
        return {
            'material': None,
            'processing_method': None,
            'feature_category': feature_category,
            'datum_reference': datum_reference,
            'geometric_tolerance_type': geo_tol_type
        }


def save_features(features: List[Node2D], output_path: str):
    """Save features to JSON file."""
    data = {
        'num_features': len(features),
        'features': [f.to_dict() for f in features]
    }
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_features(input_path: str) -> List[Node2D]:
    """Load features from JSON file."""
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    features = []
    for feat_data in data.get('features', []):
        features.append(Node2D.from_dict(feat_data))
    
    return features


def main():
    """Test 2D feature extraction."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Extract 2D features from drawings')
    parser.add_argument('-i', '--input', dest='input_path', required=True,
                       help='Input drawing file (JSON or image)')
    parser.add_argument('-o', '--output', dest='output_path',
                       help='Output features JSON file')
    parser.add_argument('--use-bert', action='store_true',
                       help='Use BERT for text embeddings')
    parser.add_argument('--use-resnet', action='store_true',
                       help='Use ResNet for visual features')
    
    args = parser.parse_args()
    
    extractor = FeatureExtractor2D({
        'use_bert': args.use_bert,
        'use_resnet': args.use_resnet
    })
    
    features = extractor.extract_features(args.input_path)
    
    print(f"Extracted {len(features)} 2D features:")
    for feature in features:
        print(f"\n  {feature.id}:")
        print(f"    Type: {feature.type}")
        print(f"    Value: {feature.value}")
        print(f"    Tolerance: {feature.tolerance}")
        print(f"    Position: {feature.position_2d}")
        print(f"    Semantic Info: {feature.semantic_info}")
        print(f"    Text: {feature.text_content}")
        print(f"    Confidence: {feature.confidence}")
        if feature.text_embedding is not None:
            print(f"    Text Embedding: {feature.text_embedding.shape}")
        if feature.visual_features is not None:
            print(f"    Visual Features: {feature.visual_features.shape}")
    
    if args.output_path:
        save_features(features, args.output_path)
        print(f"\nSaved to {args.output_path}")


if __name__ == '__main__':
    main()
