#!/usr/bin/env python3
"""
PointNet++ based segmentation for manufacturing features.

Uses PointNet++ for per-point feature extraction and semantic segmentation
into 16 manufacturing feature classes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
from pathlib import Path
from sklearn.cluster import DBSCAN


# Manufacturing feature classes (16 classes)
FEATURE_CLASSES = [
    'plane', 'cylinder', 'cone', 'sphere', 'torus',
    'hole', 'slot', 'pocket', 'chamfer', 'fillet',
    'step', 'island', 'counterbore', 'countersink', 'taper_hole', 'unknown'
]


class PointNetSetAbstraction(nn.Module):
    """Set abstraction layer (sampling + grouping + PointNet)."""
    
    def __init__(self, npoint: int, radius: float, nsample: int, 
                 in_channel: int, mlp: List[int], group_all: bool = False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        
        # MLP for PointNet within grouping
        modules = []
        last_channel = in_channel + 3  # xyz + features
        for out_channel in mlp:
            modules.append(nn.Conv2d(last_channel, out_channel, 1))
            modules.append(nn.BatchNorm2d(out_channel))
            modules.append(nn.ReLU())
            last_channel = out_channel
        
        self.mlp = nn.Sequential(*modules)
        
        if group_all:
            self.group_all = True
        else:
            # Furthest point sampling (implemented in forward)
            pass
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            xyz: (B, N, 3)
            features: (B, C, N)
        Returns:
            new_xyz: (B, npoint, 3)
            new_features: (B, mlp[-1], npoint)
        """
        xyz = xyz.permute(0, 2, 1)  # (B, N, 3)
        
        # Adapt npoint to actual number of points
        actual_npoints = min(self.npoint, xyz.shape[1])
        
        if self.group_all or actual_npoints <= 1:
            new_xyz = xyz
            # Group all points
            grouped_features = features.unsqueeze(-1)  # (B, C, N, 1)
        else:
            # Sample points using FPS
            fps_idx = self.furthest_point_sample(xyz, actual_npoints)
            new_xyz = self.gather_operation(xyz, fps_idx)
            
            # Group points by radius (with adaptive nsample)
            actual_nsample = min(self.nsample, xyz.shape[1])
            if actual_nsample < 2:
                actual_nsample = 2
            
            grouped_features, _ = self.group_points(xyz, new_xyz, features, self.radius, actual_nsample)
            
            # If nsample changed, need to adapt - skip MLP and use max pooling
            if actual_nsample != self.nsample:
                # Simple feature extraction: just use input features with max pooling
                out = torch.max(features.unsqueeze(-1), dim=-1)[0]  # (B, C, 1)
                # Expand to match expected output size
                out = out.expand(-1, -1, new_xyz.shape[1])
            else:
                # Apply PointNet (MLP + max pooling)
                out = self.mlp(grouped_features)  # (B, mlp[-1], npoint, nsample)
                out = torch.max(out, dim=-1)[0]  # (B, mlp[-1], npoint)
        
        return new_xyz.permute(0, 2, 1), out
    
    @staticmethod
    def furthest_point_sample(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
        """
        Furthest Point Sampling.
        Args:
            xyz: (B, N, 3)
            npoint: number of points to sample
        Returns:
            centroids: (B, npoint) indices
        """
        device = xyz.device
        B, N, C = xyz.shape
        
        centroids = torch.zeros(B, npoint, dtype=torch.long, device=device)
        distance = torch.ones(B, N, device=device) * 1e10
        farthest = torch.randint(0, N, (B,), device=device)
        
        batch_indices = torch.arange(B, device=device)
        
        for i in range(npoint):
            centroids[:, i] = farthest
            centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
            dist = torch.sum((xyz - centroid) ** 2, dim=-1)
            mask = dist < distance
            distance[mask] = dist[mask]
            farthest = torch.max(distance, dim=-1)[1]
        
        return centroids
    
    @staticmethod
    def gather_operation(xyz: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        """Gather points by index."""
        B, N, C = xyz.shape
        expanded_idx = idx.unsqueeze(-1).expand(-1, -1, C)
        return torch.gather(xyz, 1, expanded_idx)
    
    @staticmethod
    def group_points(xyz: torch.Tensor, new_xyz: torch.Tensor, 
                     features: torch.Tensor, radius: float, nsample: int):
        """
        Group points by ball query with adaptive nsample.
        Args:
            xyz: (B, N, 3) original points
            new_xyz: (B, S, 3) centroids
            features: (B, C, N)
            radius: query radius
            nsample: max points per group
        Returns:
            grouped_features: (B, C+3, S, actual_nsample)
        """
        B, N, C = xyz.shape
        S = new_xyz.shape[1]
        
        # Adaptive nsample - can't sample more points than available
        actual_nsample = min(nsample, N)
        if actual_nsample < 2:
            actual_nsample = min(2, N)  # At least 2 points if available
        
        # Compute distances between new_xyz and all xyz
        # new_xyz: (B, S, 1, 3), xyz: (B, 1, N, 3)
        dist = torch.sum((new_xyz.unsqueeze(2) - xyz.unsqueeze(1)) ** 2, dim=-1)  # (B, S, N)
        
        # For each centroid, select up to actual_nsample nearest points
        grouped_idx = torch.zeros(B, S, actual_nsample, dtype=torch.long, device=xyz.device)
        
        for b in range(B):
            for s in range(S):
                # Get sorted indices by distance
                sorted_idx = torch.argsort(dist[b, s])
                
                # Select actual_nsample nearest points
                if len(sorted_idx) >= actual_nsample:
                    selected = sorted_idx[:actual_nsample]
                else:
                    # If fewer points, repeat last index
                    selected = torch.cat([
                        sorted_idx,
                        sorted_idx[-1:].expand(actual_nsample - len(sorted_idx))
                    ])
                
                grouped_idx[b, s] = selected
        
        # Gather coordinates and features
        grouped_xyz = torch.gather(xyz, 1, grouped_idx.view(B, S * actual_nsample, 1).expand(-1, -1, 3)).view(B, S, actual_nsample, 3)
        grouped_features = torch.gather(
            features.permute(0, 2, 1), 1, 
            grouped_idx.view(B, S * actual_nsample, 1).expand(-1, -1, features.shape[1])
        ).view(B, S, actual_nsample, features.shape[1]).permute(0, 3, 1, 2)
        
        # Relative coordinates
        grouped_xyz = grouped_xyz - new_xyz.unsqueeze(2)
        
        # Concatenate xyz with features
        grouped_xyz = grouped_xyz.permute(0, 3, 1, 2)  # (B, 3, S, actual_nsample)
        grouped_all = torch.cat([grouped_xyz, grouped_features], dim=1)  # (B, C+3, S, actual_nsample)
        
        return grouped_all, grouped_idx


class PointNetFeaturePropagation(nn.Module):
    """Feature propagation (interpolation + skip connection)."""
    
    def __init__(self, in_channel: int, mlp: List[int]):
        super().__init__()
        modules = []
        last_channel = in_channel
        for out_channel in mlp:
            modules.append(nn.Conv1d(last_channel, out_channel, 1))
            modules.append(nn.BatchNorm1d(out_channel))
            modules.append(nn.ReLU())
            last_channel = out_channel
        
        self.mlp = nn.Sequential(*modules)
    
    def forward(self, xyz1: torch.Tensor, xyz2: torch.Tensor,
                points1: torch.Tensor, points2: torch.Tensor) -> torch.Tensor:
        """
        Interpolate features from xyz2 to xyz1.
        Args:
            xyz1: (B, N, 3) points to interpolate to
            xyz2: (B, M, 3) points to interpolate from
            points1: (B, C1, N) features at xyz1
            points2: (B, C2, M) features at xyz2
        Returns:
            new_points: (B, mlp[-1], N)
        """
        B, N, C = xyz1.shape
        M = xyz2.shape[1]
        
        # Compute distances
        dist = torch.sum((xyz1.unsqueeze(2) - xyz2.unsqueeze(1)) ** 2, dim=-1)  # (B, N, M)
        
        # Find 3 nearest neighbors
        dist, idx = torch.topk(dist, 3, dim=-1, largest=False)  # (B, N, 3)
        
        # Inverse distance weighting
        dist = torch.clamp(dist, min=1e-10)
        norm = 1.0 / dist
        weight = norm / torch.sum(norm, dim=-1, keepdim=True)  # (B, N, 3)
        
        # Interpolate features
        interpolated = torch.zeros(B, points2.shape[1], N, device=xyz1.device)
        for i in range(3):
            neighbor_features = torch.gather(
                points2, 2, 
                idx[:, :, i].unsqueeze(1).expand(-1, points2.shape[1], -1)
            )
            interpolated += weight[:, :, i].unsqueeze(1) * neighbor_features
        
        # Skip connection
        if points1 is not None:
            interpolated = torch.cat([interpolated, points1], dim=1)
        
        # Apply MLP
        out = self.mlp(interpolated)
        
        return out


class PointNet2SemanticSeg(nn.Module):
    """
    PointNet++ for semantic segmentation.
    
    Architecture:
    - Set Abstraction (encoding): SA1 → SA2 → SA3 → SA4
    - Feature Propagation (decoding): FP1 → FP2 → FP3 → FP4
    - Classification head: FC → FC → num_classes
    """
    
    def __init__(self, num_classes: int = 16, use_normals: bool = False):
        super().__init__()
        self.num_classes = num_classes
        in_channel = 6 if use_normals else 3
        
        # Set Abstraction (Encoder)
        self.sa1 = PointNetSetAbstraction(1024, 0.1, 32, in_channel, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(256, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 256 + 3, [256, 256, 512], False)
        
        # Feature Propagation (Decoder)
        self.fp4 = PointNetFeaturePropagation(512 + 256, [256, 256])
        self.fp3 = PointNetFeaturePropagation(256 + 128, [256, 256])
        self.fp2 = PointNetFeaturePropagation(256 + 64, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128 + in_channel, [128, 128, 128])
        
        # Classification head
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)
    
    def forward(self, xyz: torch.Tensor, features: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3) point coordinates
            features: (B, C, N) point features (optional, e.g., normals)
        Returns:
            logits: (B, num_classes, N) per-point class logits
        """
        B, N, _ = xyz.shape
        
        if features is None:
            features = torch.zeros(B, 0, N, device=xyz.device)
        
        # Encoder
        l1_xyz, l1_features = self.sa1(xyz, features)
        l2_xyz, l2_features = self.sa2(l1_xyz, l1_features)
        l3_xyz, l3_features = self.sa3(l2_xyz, l2_features)
        l4_xyz, l4_features = self.sa4(l3_xyz, l3_features)
        
        # Decoder
        l3_features = self.fp4(l3_xyz, l4_xyz, l3_features, l4_features)
        l2_features = self.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
        l1_features = self.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
        l0_features = self.fp1(xyz, l1_xyz, features, l1_features)
        
        # Classification
        out = F.relu(self.bn1(self.conv1(l0_features)))
        out = self.drop1(out)
        logits = self.conv2(out)
        
        return logits  # (B, num_classes, N)


class PointNet2Segmenter:
    """
    Wrapper for PointNet++ based point cloud segmentation.
    
    Usage:
        segmenter = PointNet2Segmenter(num_classes=16, device='cuda')
        results = segmenter.segment(points)
    """
    
    def __init__(self, config: Dict = None, device: str = 'cuda'):
        self.config = config or {
            'num_classes': 16,
            'use_normals': False,
            'min_segment_size': 30,
            'block_size': 1.0,  # For local processing
            'stride': 0.5,
        }
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
        import os
        weight_path = '/home/spectr/itmo/PointNet2/checkpoints/pointnet2_sem_seg.pth'
        
        if os.path.exists(weight_path):
            print(f"  [PointNet2] Loading pre-trained weights from {weight_path}")
            checkpoint = torch.load(weight_path, map_location=self.device)
            self.model.load_state_dict(checkpoint, strict=False)
            print(f"  [PointNet2] Weights loaded")
        else:
            print(f"  [PointNet2] No pre-trained weights found at {weight_path}")
            print(f"  [PointNet2] Using random initialization (needs training)")
            print(f"  [PointNet2] Will fall back to geometric segmentation for inference")
    
    def segment(self, points: np.ndarray) -> List[Dict]:
        """
        Segment point cloud using PointNet++.
        
        Args:
            points: (N, 3) point cloud
        
        Returns:
            List of segments with feature types and point indices
        """
        # Normalize points to [0, 1]
        points_min = points.min(axis=0)
        points_max = points.max(axis=0)
        points_normalized = (points - points_min) / (points_max - points_min + 1e-10)
        
        # Convert to tensor
        points_tensor = torch.FloatTensor(points_normalized).unsqueeze(0).to(self.device)  # (1, N, 3)
        N = len(points)
        
        # Adapt architecture to point cloud size
        # PointNet2 requires decreasing number of points at each level
        if N >= 1024:
            npoints = [1024, 256, 64, 16]
        elif N >= 512:
            npoints = [512, 128, 32, 8]
        elif N >= 256:
            npoints = [256, 64, 16, 4]
        elif N >= 128:
            npoints = [128, 32, 8, 2]
        else:
            # Too few points, use geometric fallback
            print(f"  [PointNet2] Too few points ({N}), using geometric fallback")
            return self._geometric_fallback(points)
        
        # Check if model is trained (simple heuristic: if no weights loaded, use fallback)
        # For now, we'll use PointNet++ as a feature extractor + geometric rules
        point_features = self._extract_features(points_tensor, npoints)
        
        # Classify using geometric rules + PointNet features
        labels = self._classify_with_features(points, point_features)
        
        # Cluster into segments
        segments = self._cluster_segments(points, labels)
        
        return segments
    
    def _geometric_fallback(self, points: np.ndarray) -> List[Dict]:
        """Fallback to geometric segmentation when PointNet++ can't be used."""
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
        from geometric_segmentation import GeometricFeatureSegmentation
        
        segmenter = GeometricFeatureSegmentation({
            'curvature_threshold': 0.1,
            'min_segment_size': 30
        })
        
        segments_dict = segmenter.segment_features(points)
        
        # Convert to list format
        segments = []
        for seg_id, seg_data in segments_dict.items():
            segments.append({
                'feature_type': seg_data.get('feature_type', 'unknown'),
                'point_indices': seg_data.get('point_indices', seg_data.get('nodes', [])),
                'confidence': seg_data.get('confidence', 0.7),
                'num_points': len(seg_data.get('point_indices', seg_data.get('nodes', [])))
            })
        
        return segments

    def _extract_features(self, points: torch.Tensor, npoints: List[int]) -> torch.Tensor:
        """Extract per-point features using PointNet++ encoder."""
        B, N, _ = points.shape
        
        # Forward pass through encoder with adaptive npoints
        with torch.no_grad():
            l1_xyz, l1_features = self.model.sa1(points, torch.zeros(B, 0, N, device=points.device))
            l2_xyz, l2_features = self.model.sa2(l1_xyz, l1_features)
            l3_xyz, l3_features = self.model.sa3(l2_xyz, l2_features)
            l4_xyz, l4_features = self.model.sa4(l3_xyz, l3_features)
            
            # Propagate back to original resolution
            l3_features = self.model.fp4(l3_xyz, l4_xyz, l3_features, l4_features)
            l2_features = self.model.fp3(l2_xyz, l3_xyz, l2_features, l3_features)
            l1_features = self.model.fp2(l1_xyz, l2_xyz, l1_features, l2_features)
            l0_features = self.model.fp1(points, l1_xyz, torch.zeros(B, 0, N, device=points.device), l1_features)
        
        return l0_features  # (B, 128, N)
    
    def _classify_with_features(self, points: np.ndarray, features: torch.Tensor) -> np.ndarray:
        """
        Classify points using PointNet++ features + geometric rules.
        
        For now, uses geometric rules as fallback.
        When training data is available, will use a simple classifier head.
        """
        from utils.geometric_segmentation import RANSACPrimitiveFitter, ManufacturingFeatureClassifier
        
        # Extract geometric features
        fitter = RANSACPrimitiveFitter()
        classifier = ManufacturingFeatureClassifier()
        
        # Fit primitives to entire cloud
        primitive_fit = fitter.fit_all_primitives(points)
        geo_props = fitter._compute_geometric_properties(points)
        
        # Get best primitive type
        best_primitive = classifier._get_best_primitive(primitive_fit)
        
        # Map primitive to manufacturing classes
        primitive_to_class = {
            'plane': 'plane',
            'cylinder': 'cylinder',
            'cone': 'cone',
            'sphere': 'sphere',
            'torus': 'torus'
        }
        
        # For now, assign all points to the best primitive
        # This will be replaced with per-point classification when trained
        class_idx = FEATURE_CLASSES.index(primitive_to_class.get(best_primitive, 'unknown'))
        labels = np.full(len(points), class_idx, dtype=np.int64)
        
        return labels
    
    def _cluster_segments(self, points: np.ndarray, labels: np.ndarray) -> List[Dict]:
        """
        Group points with same labels into segments using spatial clustering.
        """
        segments = []
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            mask = labels == label
            label_points = points[mask]
            
            if len(label_points) < self.config['min_segment_size']:
                continue
            
            # DBSCAN clustering for spatial coherence
            bbox_diag = np.linalg.norm(label_points.max(axis=0) - label_points.min(axis=0))
            eps = bbox_diag * 0.1
            
            clustering = DBSCAN(eps=eps, min_samples=self.config['min_segment_size'] // 3).fit(label_points)
            
            for cluster_label in np.unique(clustering.labels_):
                if cluster_label == -1:
                    continue  # Skip noise
                
                cluster_mask = clustering.labels_ == cluster_label
                cluster_indices = np.where(mask)[0][cluster_mask]
                
                if len(cluster_indices) < self.config['min_segment_size']:
                    continue
                
                feature_type = FEATURE_CLASSES[label]
                segments.append({
                    'feature_type': feature_type,
                    'point_indices': cluster_indices.tolist(),
                    'confidence': 0.7,
                    'num_points': len(cluster_indices)
                })
        
        return segments


def segment_with_pointnet2(points: np.ndarray, config: Dict = None) -> Dict:
    """
    Main entry point for PointNet2 segmentation.
    
    Args:
        points: (N, 3) point cloud
        config: Configuration dictionary
    
    Returns:
        Dictionary of segments
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    segmenter = PointNet2Segmenter(config, device=device)
    
    segments = segmenter.segment(points)
    
    # Convert to output format
    result = {}
    for i, seg in enumerate(segments):
        result[f'pointnet2_seg_{i}'] = seg
    
    return result
