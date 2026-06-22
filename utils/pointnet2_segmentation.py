#!/usr/bin/env python3
"""
PointNet++ based semantic segmentation for manufacturing features.

Implements per-point classification into 16 manufacturing feature classes:
- plane, cylinder, cone, sphere, torus (geometric primitives)
- hole, slot, pocket, chamfer, fillet, step, island, counterbore, countersink, taper_hole (manufacturing features)
- unknown (unclassified)

When trained weights are available, uses PointNet++ for segmentation.
Otherwise, falls back to RANSAC + DBSCAN for local segmentation.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from dataclasses import dataclass, field


# Manufacturing feature classes (16 classes)
FEATURE_CLASSES = [
    'plane', 'cylinder', 'cone', 'sphere', 'torus',
    'hole', 'slot', 'pocket', 'chamfer', 'fillet',
    'step', 'island', 'counterbore', 'countersink', 'taper_hole', 'unknown'
]


@dataclass
class SegmentResult:
    """Result of PointNet++ segmentation."""
    segment_id: str
    feature_type: str
    point_indices: List[int]
    confidence: float
    geometric_params: Dict = field(default_factory=dict)
    local_measurements: Dict = field(default_factory=dict)


class PointNet2SemanticSeg(nn.Module):
    """
    PointNet++ for semantic segmentation.
    
    Architecture:
    - Set Abstraction (SA): PointNet++ encoder
    - Feature Propagation (FP): PointNet++ decoder
    - Classification head: per-point classification
    """
    
    def __init__(self, num_classes: int = 16, use_normals: bool = False):
        super().__init__()
        self.num_classes = num_classes
        in_channel = 6 if use_normals else 3
        
        # Encoder layers (simplified PointNet++)
        self.conv1 = nn.Conv1d(in_channel, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = nn.Conv1d(128, 256, 1)
        self.bn3 = nn.BatchNorm1d(256)
        self.conv4 = nn.Conv1d(256, 512, 1)
        self.bn4 = nn.BatchNorm1d(512)
        
        # Decoder layers
        self.upconv1 = nn.Conv1d(512, 256, 1)
        self.upbn1 = nn.BatchNorm1d(256)
        self.upconv2 = nn.Conv1d(256, 128, 1)
        self.upbn2 = nn.BatchNorm1d(128)
        
        # Classification head
        self.conv5 = nn.Conv1d(128, 64, 1)
        self.bn5 = nn.BatchNorm1d(64)
        self.dropout = nn.Dropout(0.5)
        self.conv6 = nn.Conv1d(64, num_classes, 1)
        
        # FPS and grouping helpers
        self.use_normals = use_normals
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, C, N) additional features (e.g., normals)
        Returns:
            logits: (B, num_classes, N) per-point logits
        """
        B, N, _ = xyz.shape
        
        # Prepare input
        if features is not None:
            x = torch.cat([xyz.permute(0, 2, 1), features], dim=1)
        else:
            x = xyz.permute(0, 2, 1)
        
        # Encoder
        x1 = F.relu(self.bn1(self.conv1(x)))
        x2 = F.relu(self.bn2(self.conv2(x1)))
        x3 = F.relu(self.bn3(self.conv3(x2)))
        x4 = F.relu(self.bn4(self.conv4(x3)))
        
        # Global feature
        global_feat = torch.max(x4, dim=2, keepdim=True)[0]
        global_feat = global_feat.expand(-1, -1, N)
        
        # Decoder with skip connections
        x = torch.cat([x3, global_feat], dim=1)
        x = F.relu(self.upbn1(self.upconv1(x)))
        x = torch.cat([x2, x], dim=1)
        x = F.relu(self.upbn2(self.upconv2(x)))
        x = torch.cat([x1, x], dim=1)
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.dropout(x)
        logits = self.conv6(x)
        
        return logits


class PointNet2Segmenter:
    """
    PointNet++ based segmenter with local measurements.
    
    Usage:
        segmenter = PointNet2Segmenter(device='cuda')
        segments = segmenter.segment(points)
    """
    
    def __init__(self, config: Dict = None, device: str = 'cuda'):
        self.config = config or {}
        self.config.setdefault('num_classes', 16)
        self.config.setdefault('use_normals', False)
        self.config.setdefault('min_segment_size', 30)
        
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_classes = self.config['num_classes']
        
        # Build model
        self.model = PointNet2SemanticSeg(
            num_classes=self.num_classes,
            use_normals=self.config['use_normals']
        )
        self.model.to(self.device)
        self.model.eval()
        
        # Try to load pre-trained weights
        self._load_weights()
    
    def _load_weights(self):
        """Try to load pre-trained weights."""
        weight_paths = [
            '/home/spectr/itmo/PointNet2/checkpoints/pointnet2_sem_seg.pth',
            '/home/spectr/itmo/PointNet2/checkpoints/best_model.pth',
            '/home/spectr/itmo/checkpoints/pointnet2_manufacturing.pth',
        ]
        
        for path in weight_paths:
            if Path(path).exists():
                print(f"  [PointNet2] Loading weights from {path}")
                try:
                    checkpoint = torch.load(path, map_location=self.device)
                    if 'model_state_dict' in checkpoint:
                        self.model.load_state_dict(checkpoint['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint, strict=False)
                    self.is_trained = True
                    print(f"  [PointNet2] Weights loaded successfully")
                    return
                except Exception as e:
                    print(f"  [PointNet2] Failed to load {path}: {e}")
        
        print(f"  [PointNet2] No pre-trained weights found, using geometric fallback")
        self.is_trained = False
    
    def segment(self, points: np.ndarray) -> List[SegmentResult]:
        """
        Segment point cloud using PointNet++.
        
        Args:
            points: (N, 3) point cloud
        
        Returns:
            List of SegmentResult objects
        """
        if self.is_trained:
            return self._segment_with_pointnet2(points)
        else:
            print(f"  [PointNet2] Using geometric segmentation fallback")
            return self._segment_with_geometry(points)
    
    def _segment_with_pointnet2(self, points: np.ndarray) -> List[SegmentResult]:
        """Segment using trained PointNet++ model."""
        # Normalize points
        points_min = points.min(axis=0)
        points_max = points.max(axis=0)
        points_norm = (points - points_min) / (points_max - points_min + 1e-10)
        
        # Convert to tensor
        points_tensor = torch.FloatTensor(points_norm).unsqueeze(0).to(self.device)
        
        # Forward pass
        with torch.no_grad():
            logits = self.model(points_tensor)
            probs = F.softmax(logits, dim=1)
            labels = torch.argmax(probs, dim=1).squeeze().cpu().numpy()
        
        # Group points by label
        segments = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            point_indices = np.where(mask)[0]
            
            if len(point_indices) < self.config['min_segment_size']:
                continue
            
            feature_type = FEATURE_CLASSES[label]
            confidence = float(probs[0, label, mask].mean())
            
            # Compute local measurements
            seg_points = points[mask]
            measurements = self._compute_local_measurements(seg_points, feature_type)
            
            # Get geometric parameters
            geo_params = self._fit_local_primitive(seg_points, feature_type)
            
            segment = SegmentResult(
                segment_id=f'pointnet2_{label}_{len(segments)}',
                feature_type=feature_type,
                point_indices=point_indices.tolist(),
                confidence=confidence,
                geometric_params=geo_params,
                local_measurements=measurements
            )
            
            segments.append(segment)
        
        return segments
    
    def _segment_with_geometry(self, points: np.ndarray) -> List[SegmentResult]:
        """
        Fallback: Use hierarchical segmentation with local measurements.
        
        DBSCAN doesn't work well on sparse clouds, so we use the original
        RANSAC-based hierarchical segmentation with per-segment measurements.
        """
        from geometric_segmentation import HierarchicalSegmenter
        
        print(f"  [Geometric] Using hierarchical segmentation")
        
        # Use hierarchical segmenter
        segmenter = HierarchicalSegmenter({
            'min_segment_size': max(10, self.config['min_segment_size'] // 3),
            'curvature_threshold': 0.1
        })
        
        segments_raw = segmenter.segment(points)
        
        # Convert to our format with local measurements
        segments = []
        for seg in segments_raw:
            seg_points = points[seg.point_indices]
            
            # Compute local measurements
            measurements = self._compute_local_measurements(seg_points, seg.feature_type)
            
            # Fit primitive for additional params
            geo_params = self._fit_local_primitive(seg_points, seg.feature_type)
            
            # Merge geometric params
            geo_params.update(seg.geometric_params)
            
            segment = SegmentResult(
                segment_id=seg.segment_id,
                feature_type=seg.feature_type,
                point_indices=seg.point_indices,
                confidence=seg.confidence,
                geometric_params=geo_params,
                local_measurements=measurements
            )
            
            segments.append(segment)
        
        print(f"  [Geometric] Segmented into {len(segments)} segments")
        for seg in segments:
            print(f"    {seg.segment_id}: {seg.feature_type}, {len(seg.point_indices)} points")
            meas = seg.local_measurements
            if meas.get('diameter_mm'):
                print(f"      diameter: {meas['diameter_mm']:.1f}mm")
            if meas.get('height_mm'):
                print(f"      height: {meas['height_mm']:.1f}mm")
            if meas.get('depth_mm'):
                print(f"      depth: {meas['depth_mm']:.1f}mm")
        
        return segments
    
    def _fit_local_primitive(self, points: np.ndarray, feature_type: str = None) -> Dict:
        """
        Fit geometric primitive to local point cluster.
        
        Returns dict with feature_type, confidence, and primitive parameters.
        """
        if len(points) < 10:
            return {'feature_type': 'unknown', 'confidence': 0.3}
        
        # Compute centroid and centered points
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        
        # PCA analysis
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Sort eigenvalues
        idx = np.argsort(eigvals)[::-1]
        eigvals = eigvals[idx]
        eigvecs = eigvecs[:, idx]
        
        # Compute shape measures
        planarity = (eigvals[1] - eigvals[0]) / (eigvals[0] + 1e-10)
        linearity = (eigvals[0] - eigvals[1]) / (eigvals[0] + 1e-10)
        sphericity = eigvals[2] / (eigvals[0] + 1e-10)
        
        # Bounding box dimensions (points already in mm)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        sorted_sizes = np.sort(bbox_size)[::-1]
        
        # Diameter (max extent perpendicular to principal axis)
        if len(sorted_sizes) >= 2:
            diameter = sorted_sizes[1]  # Already in mm
        else:
            diameter = 0.0
        
        # Height (extent along principal axis)
        if len(sorted_sizes) >= 3:
            height = sorted_sizes[0]  # Already in mm
        else:
            height = sorted_sizes[-1] * 1000.0 if len(sorted_sizes) > 0 else 0.0
        
        # Radius estimates
        radius_xy = np.std(points[:, 0]) * 2  # Rough estimate
        radius = np.linalg.norm(points[:, :2], axis=1).mean()
        
        # Classification based on shape measures
        if feature_type is None:
            if planarity > 5.0 and eigvals[0] < 0.01:
                feature_type = 'plane'
                confidence = min(0.9, planarity / 10.0)
            elif sphericity > 0.5 and abs(eigvals[0] - eigvals[2]) < 0.1:
                feature_type = 'sphere'
                confidence = min(0.85, sphericity)
            elif linearity > 0.5:
                feature_type = 'cylinder'
                confidence = min(0.8, linearity)
            elif 1.0 < planarity < 5.0:
                feature_type = 'cone'
                confidence = 0.6
            else:
                # Manufacturing feature classification
                aspect_ratio = sorted_sizes[1] / (sorted_sizes[0] + 1e-10)
                
                if aspect_ratio > 0.8 and sorted_sizes[2] < 0.1:
                    feature_type = 'hole'
                    confidence = 0.7
                elif aspect_ratio < 0.3:
                    feature_type = 'slot'
                    confidence = 0.65
                elif sorted_sizes[2] < 0.2:
                    feature_type = 'pocket'
                    confidence = 0.6
                else:
                    feature_type = 'unknown'
                    confidence = 0.3
        else:
            confidence = 0.7  # Default for externally specified type
        
        return {
            'feature_type': feature_type,
            'confidence': confidence,
            'centroid': centroid.tolist(),
            'principal_directions': eigvecs.tolist(),
            'eigenvalues': eigvals.tolist(),
            'bounding_box': {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist(),
                'sizes': bbox_size.tolist()
            },
            'diameter': diameter,  # mm
            'height': height,      # mm
            'radius': radius,      # mm
            'planarity': float(planarity),
            'linearity': float(linearity),
            'sphericity': float(sphericity)
        }
    
    def _compute_local_measurements(self, points: np.ndarray, feature_type: str) -> Dict:
        """
        Compute local geometric measurements for a segment.
        
        Points are assumed to be in millimeters (mm).
        Returns measurements in mm.
        """
        if len(points) < 3:
            return {}
        
        # Bounding box (points already in mm)
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        # Sort dimensions
        sorted_dims = np.sort(bbox_size)[::-1]
        
        measurements = {
            'bounding_box_mm': {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist(),
                'sizes': bbox_size.tolist()
            },
            'principal_dimensions_mm': {
                'length': float(sorted_dims[0]),
                'width': float(sorted_dims[1]) if len(sorted_dims) > 1 else 0.0,
                'thickness': float(sorted_dims[2]) if len(sorted_dims) > 2 else 0.0
            }
        }
        
        # Feature-specific measurements
        if feature_type in ['cylinder', 'hole', 'counterbore', 'countersink']:
            # Diameter and height
            measurements['diameter_mm'] = float(sorted_dims[1]) if len(sorted_dims) > 1 else 0.0
            measurements['height_mm'] = float(sorted_dims[0]) if len(sorted_dims) > 0 else 0.0
            measurements['depth_mm'] = measurements['height_mm']
            
        elif feature_type in ['cone', 'taper_hole']:
            # Base diameter and height
            measurements['base_diameter_mm'] = float(sorted_dims[1]) if len(sorted_dims) > 1 else 0.0
            measurements['height_mm'] = float(sorted_dims[0]) if len(sorted_dims) > 0 else 0.0
            measurements['depth_mm'] = measurements['height_mm']
            
        elif feature_type in ['sphere']:
            # Radius
            center = np.mean(points, axis=0)
            radii = np.linalg.norm(points - center, axis=1)
            measurements['radius_mm'] = float(np.mean(radii))
            measurements['diameter_mm'] = measurements['radius_mm'] * 2
            
        elif feature_type in ['plane']:
            # Area and thickness
            measurements['area_mm2'] = float(sorted_dims[0] * sorted_dims[1])
            measurements['thickness_mm'] = float(sorted_dims[2]) if len(sorted_dims) > 2 else 0.0
            
        elif feature_type in ['slot', 'pocket']:
            measurements['length_mm'] = float(sorted_dims[0])
            measurements['width_mm'] = float(sorted_dims[1]) if len(sorted_dims) > 1 else 0.0
            measurements['depth_mm'] = float(sorted_dims[2]) if len(sorted_dims) > 2 else 0.0
        
        return measurements


def segment_with_pointnet2(points: np.ndarray, config: Dict = None) -> List[SegmentResult]:
    """
    Main entry point for PointNet2 segmentation.
    
    Args:
        points: (N, 3) point cloud
        config: Configuration dictionary
    
    Returns:
        List of SegmentResult objects
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    segmenter = PointNet2Segmenter(config, device=device)
    
    return segmenter.segment(points)
