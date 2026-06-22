#!/usr/bin/env python3
"""
Geometric Feature Segmentation for 3D Point Clouds.

Implements RANSAC-based segmentation for manufacturing features:
- 16 classes: hole, slot, pocket, chamfer, fillet, step, island, counterbore, 
  countersink, taper_hole, plane, cylinder, cone, sphere, torus, unknown
- 3-level hierarchical segmentation
- Geometric primitive fitting (plane, cylinder, cone, sphere, torus)
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field


# Manufacturing feature classes
FEATURE_CLASSES = [
    'hole', 'slot', 'pocket', 'chamfer', 'fillet', 'step', 'island',
    'counterbore', 'countersink', 'taper_hole', 'plane', 'cylinder',
    'cone', 'sphere', 'torus', 'unknown'
]


@dataclass
class SegmentResult:
    """Result of geometric segmentation."""
    segment_id: str
    feature_type: str
    point_indices: List[int]
    confidence: float
    geometric_params: Dict = field(default_factory=dict)
    hierarchy_level: int = 0
    parent_id: Optional[str] = None
    children_ids: List[str] = field(default_factory=list)


class RANSACPrimitiveFitter:
    """
    Fit geometric primitives to point clouds using RANSAC.
    
    Supports: plane, cylinder, cone, sphere, torus
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_iterations': 1000,
            'inlier_threshold': 0.01,  # 1% of bounding box diagonal
            'min_inlier_ratio': 0.3,   # Minimum 30% inliers
            'num_samples_plane': 3,
            'num_samples_cylinder': 5,
            'num_samples_sphere': 4,
        }
    
    def fit_all_primitives(self, points: np.ndarray) -> Dict:
        """Try to fit all primitives and return best results."""
        results = {}
        
        # Compute adaptive threshold
        bbox_diag = np.linalg.norm(points.max(axis=0) - points.min(axis=0))
        threshold = self.config['inlier_threshold'] * bbox_diag
        
        # Fit each primitive
        results['plane'] = self._fit_plane(points, threshold)
        results['cylinder'] = self._fit_cylinder(points, threshold)
        results['sphere'] = self._fit_sphere(points, threshold)
        results['cone'] = self._fit_cone(points, threshold)
        results['torus'] = self._fit_torus(points, threshold)
        
        return results
    
    def _fit_plane(self, points: np.ndarray, threshold: float) -> Dict:
        """Fit plane using SVD."""
        if len(points) < 3:
            return {'inliers': [], 'inlier_ratio': 0, 'params': {}}
        
        # Fit plane using SVD
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        _, _, vh = np.linalg.svd(centered)
        normal = vh[-1]  # Last singular vector is normal
        
        # Compute distances to plane
        distances = np.abs(np.dot(centered, normal))
        inlier_mask = distances < threshold
        inlier_ratio = np.sum(inlier_mask) / len(points)
        
        return {
            'inliers': np.where(inlier_mask)[0].tolist(),
            'inlier_ratio': inlier_ratio,
            'params': {
                'centroid': centroid.tolist(),
                'normal': normal.tolist(),
                'residual': float(np.mean(distances[inlier_mask])) if np.sum(inlier_mask) > 0 else 999
            }
        }
    
    def _fit_cylinder(self, points: np.ndarray, threshold: float) -> Dict:
        """Fit cylinder using PCA + optimization."""
        if len(points) < 5:
            return {'inliers': [], 'inlier_ratio': 0, 'params': {}}

        # PCA to find cylinder axis
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)

        # Smallest eigenvalue direction is cylinder axis
        axis = eigvecs[:, 0]
        axis = axis / np.linalg.norm(axis)

        # Project points onto plane perpendicular to axis
        projections = centered - np.outer(np.dot(centered, axis), axis)
        radii = np.linalg.norm(projections, axis=1)

        # Fit cylinder radius
        radius = np.mean(radii)

        # Compute distances to cylinder surface
        distances = np.abs(radii - radius)
        inlier_mask = distances < threshold
        inlier_ratio = np.sum(inlier_mask) / len(points)

        # Compute cylinder height from axis projection
        axis_projections = np.dot(centered, axis)
        height = np.max(axis_projections) - np.min(axis_projections)

        return {
            'inliers': np.where(inlier_mask)[0].tolist(),
            'inlier_ratio': inlier_ratio,
            'params': {
                'centroid': centroid.tolist(),
                'axis': axis.tolist(),
                'radius': float(radius),
                'height': float(height),
                'residual': float(np.mean(distances[inlier_mask])) if np.sum(inlier_mask) > 0 else 999,
                'diameter': float(radius * 2)
            }
        }

    def _fit_sphere(self, points: np.ndarray, threshold: float) -> Dict:
        """Fit sphere using least squares."""
        if len(points) < 4:
            return {'inliers': [], 'inlier_ratio': 0, 'params': {}}
        
        # Algebraic sphere fit
        A = np.hstack([2 * points, np.ones((len(points), 1))])
        b = np.sum(points ** 2, axis=1)
        
        try:
            x = np.linalg.lstsq(A, b, rcond=None)[0]
            center = x[:3]
            radius = np.sqrt(x[3] + np.sum(center ** 2))
            
            # Compute distances to sphere
            distances = np.abs(np.linalg.norm(points - center, axis=1) - radius)
            inlier_mask = distances < threshold
            inlier_ratio = np.sum(inlier_mask) / len(points)
            
            return {
                'inliers': np.where(inlier_mask)[0].tolist(),
                'inlier_ratio': inlier_ratio,
                'params': {
                    'center': center.tolist(),
                    'radius': float(radius),
                    'residual': float(np.mean(distances[inlier_mask])) if np.sum(inlier_mask) > 0 else 999
                }
            }
        except:
            return {'inliers': [], 'inlier_ratio': 0, 'params': {}}
    
    def _fit_cone(self, points: np.ndarray, threshold: float) -> Dict:
        """Fit cone (simplified - uses cylinder as approximation)."""
        if len(points) < 5:
            return {'inliers': [], 'inlier_ratio': 0, 'params': {}}
        
        # For now, use cylinder fit as approximation
        # Full cone fitting requires more complex optimization
        cyl_result = self._fit_cylinder(points, threshold * 1.5)
        cyl_result['params']['cone_angle'] = 0.0  # Placeholder
        
        return cyl_result
    
    def _fit_torus(self, points: np.ndarray, threshold: float) -> Dict:
        """Fit torus (simplified)."""
        if len(points) < 10:
            return {'inliers': [], 'inlier_ratio': 0, 'params': {}}
        
        # Simplified: try to fit as ring of points
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        major_radius = np.mean(distances)
        
        # Check if points form ring-like structure
        radii_variation = np.std(distances) / (major_radius + 1e-10)
        
        if radii_variation < 0.3:  # Relatively uniform ring
            inlier_mask = np.abs(distances - major_radius) < threshold * 2
            inlier_ratio = np.sum(inlier_mask) / len(points)
            
            return {
                'inliers': np.where(inlier_mask)[0].tolist(),
                'inlier_ratio': inlier_ratio,
                'params': {
                    'center': centroid.tolist(),
                    'major_radius': float(major_radius),
                    'minor_radius': float(radii_variation * major_radius),
                    'residual': float(radii_variation)
                }
            }
        
        return {'inliers': [], 'inlier_ratio': 0, 'params': {}}


class ManufacturingFeatureClassifier:
    """
    Classify point cloud segments into manufacturing feature types.
    
    16 classes: hole, slot, pocket, chamfer, fillet, step, island,
                counterbore, countersink, taper_hole, plane, cylinder,
                cone, sphere, torus, unknown
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'hole_aspect_ratio': 0.8,
            'slot_aspect_ratio': 0.3,
            'chamfer_angle_range': (30, 60),  # degrees
            'fillet_curvature_threshold': 0.1,
        }
    
    def classify(self, points: np.ndarray, primitive_fit: Dict,
                 geometric_props: Dict) -> Tuple[str, float]:
        """
        Classify segment into manufacturing feature type.

        Uses PCA-based shape measures + bounding box ratios for robust classification.
        """
        # Get best fitting primitive
        best_primitive = self._get_best_primitive(primitive_fit)

        # Extract geometric characteristics
        curvature = geometric_props.get('mean_curvature', 0)
        planarity = geometric_props.get('planarity', 0)
        circularity = geometric_props.get('circularity', 0)
        aspect_ratio = geometric_props.get('aspect_ratio', 1)
        bbox = geometric_props.get('bounding_box', None)
        
        # Debug
        import os
        if os.environ.get('DEBUG_SEGMENT'):
            print(f"  [Classify] best_primitive={best_primitive}")
            print(f"    curvature={curvature:.3f}, planarity={planarity:.3f}")
            print(f"    circularity={circularity:.3f}, aspect_ratio={aspect_ratio:.3f}")
            if bbox and 'sizes' in bbox:
                sizes = sorted(bbox['sizes'], reverse=True)
                thickness_ratio = sizes[2] / (sizes[1] + 1e-10) if len(sizes) >= 3 else 0
                length_ratio = sizes[0] / (sizes[1] + 1e-10) if len(sizes) >= 2 else 1
                print(f"    thickness_ratio={thickness_ratio:.3f}, length_ratio={length_ratio:.3f}")
        
        # Compute bounding box ratios for better classification
        if bbox and 'sizes' in bbox:
            sizes = sorted(bbox['sizes'], reverse=True)
            # Thickness-to-width ratio (hole if ~1.0, plate if ~0)
            thickness_ratio = sizes[2] / (sizes[1] + 1e-10) if len(sizes) >= 3 else 0
            # Length-to-width ratio (cylinder if >2, hole if ~1)
            length_ratio = sizes[0] / (sizes[1] + 1e-10) if len(sizes) >= 2 else 1
        else:
            thickness_ratio = 0
            length_ratio = 1

        # Hole detection: short cylinder with thickness/width ratio ~1
        if best_primitive == 'cylinder':
            # Debug
            import os
            if os.environ.get('DEBUG_SEGMENT'):
                print(f"  [Classify] cylinder: thickness_ratio={thickness_ratio:.3f}, length_ratio={length_ratio:.3f}")
                print(f"    circularity={circularity:.3f}, aspect_ratio={aspect_ratio:.3f}")
            
            if thickness_ratio > 0.7 and length_ratio < 2.0:
                # Short, wide cylinder = hole
                return 'hole', 0.85
            elif length_ratio > 2:
                # Long, thin cylinder = cylinder
                return 'cylinder', 0.85
            elif circularity > 0.8 and aspect_ratio < 0.5:
                return 'hole', 0.8
            else:
                return 'cylinder', 0.7

        elif best_primitive == 'plane' and planarity > 0.8:
            return 'plane', 0.9

        elif best_primitive == 'sphere':
            # Sphere or hole (short cylinder can look like sphere)
            if circularity > 0.6:  # Lowered from 0.7 for sparse clouds
                # Check if it's actually a hole (short cylinder)
                if 'sizes' in (bbox or {}):
                    sizes = sorted(bbox['sizes'], reverse=True)
                    thickness_ratio = sizes[2] / (sizes[1] + 1e-10) if len(sizes) >= 3 else 0
                    if thickness_ratio > 0.6 and length_ratio < 1.5:
                        return 'hole', 0.75
                return 'sphere', 0.80
            else:
                return 'unknown', 0.4

        elif best_primitive == 'cone':
            return 'cone', 0.7

        elif best_primitive == 'torus':
            return 'torus', 0.7

        # Manufacturing-specific classification based on shape
        if curvature > 0.3 and thickness_ratio < 0.5:
            if circularity > 0.6:
                return 'hole', 0.75
            else:
                return 'pocket', 0.7

        elif curvature > 0.2 and 0.3 < aspect_ratio < 0.7:
            return 'slot', 0.7

        elif curvature > 0.15 and length_ratio > 2:
            return 'fillet', 0.65

        elif planarity > 0.6 and length_ratio > 1.5:
            return 'step', 0.6

        elif curvature < 0.05 and planarity > 0.5:
            return 'plane', 0.7

        # Default
        return 'unknown', 0.3
    
    def _get_best_primitive(self, primitive_fit: Dict) -> str:
        """Get best fitting primitive based on inlier ratio."""
        best_name = 'unknown'
        best_ratio = 0

        for name, result in primitive_fit.items():
            ratio = result.get('inlier_ratio', 0)
            if ratio > best_ratio:
                best_ratio = ratio
                best_name = name
        
        # Debug
        import os
        if os.environ.get('DEBUG_SEGMENT'):
            print(f"  [_get_best_primitive] All primitives: ")
            for name, result in primitive_fit.items():
                print(f"    {name}: inlier_ratio={result.get('inlier_ratio', 0):.3f}")
            print(f"    → Best: {best_name} ({best_ratio:.3f})")

        return best_name


class HierarchicalSegmenter:
    """
    3-level hierarchical segmentation of point clouds.
    
    Level 0: Coarse segmentation (major features: plane, cylinder, sphere)
    Level 1: Medium segmentation (manufacturing features: hole, slot, pocket)
    Level 2: Fine segmentation (sub-features: chamfer, fillet, step)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'curvature_threshold': 0.1,
            'min_segment_size': 30,
            'max_levels': 3
        }
        self.primitive_fitter = RANSACPrimitiveFitter()
        self.classifier = ManufacturingFeatureClassifier()
    
    def segment(self, points: np.ndarray) -> List[SegmentResult]:
        """
        Perform hierarchical segmentation.

        Args:
            points: Input point cloud (N, 3)

        Returns:
            List of SegmentResult objects
        """
        segments = []

        # Step 1: Spatial clustering with DBSCAN to get local segments
        spatial_clusters = self._spatial_pre_clustering(points)
        
        # Step 2: For each spatial cluster, fit primitives and classify
        for cluster_idx, cluster_indices in enumerate(spatial_clusters):
            if len(cluster_indices) < self.config['min_segment_size']:
                continue
            
            cluster_points = points[cluster_indices]
            
            # Fit primitives to this local cluster
            primitive_fit = self.primitive_fitter.fit_all_primitives(cluster_points)
            geo_props = self._compute_geometric_properties(cluster_points)
            
            # Classify
            feature_type, confidence = self.classifier.classify(
                cluster_points, primitive_fit, geo_props
            )
            
            # Create segment
            segment = SegmentResult(
                segment_id=f'segment_{cluster_idx}',
                feature_type=feature_type,
                point_indices=cluster_indices.tolist(),
                confidence=confidence,
                geometric_params=primitive_fit,
                hierarchy_level=0
            )
            
            segments.append(segment)

        # If DBSCAN didn't produce good segments, fall back to hierarchical
        if not segments or all(len(s.point_indices) < self.config['min_segment_size'] for s in segments):
            segments = self._fallback_hierarchical_segmentation(points)

        return segments
    
    def _spatial_pre_clustering(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Pre-cluster points using DBSCAN spatial clustering.
        
        Uses average point spacing for adaptive eps to handle both
        large sparse clouds and small dense clouds correctly.
        """
        from sklearn.cluster import DBSCAN
        from scipy.spatial import KDTree

        # Calculate average point spacing using k-NN
        k = min(10, len(points) - 1)
        tree = KDTree(points)
        distances, _ = tree.query(points, k=k+1)
        
        # Average distance to k-th nearest neighbor
        avg_spacing = np.mean(distances[:, -1])
        
        # eps = 3-4x average spacing for meaningful clusters
        eps = avg_spacing * 3.5
        
        # Ensure reasonable bounds (but not too restrictive)
        eps = max(eps, 2.0)    # At least 2mm for very small features
        eps = min(eps, 100.0)  # At most 100mm
        
        # DBSCAN clustering
        min_samples = max(5, k)
        clustering = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric='euclidean',
            n_jobs=-1
        ).fit(points)

        labels = clustering.labels_
        
        # Debug info
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        n_noise = list(labels).count(-1)
        print(f"  [DBSCAN] eps={eps:.1f}mm (avg_spacing={avg_spacing:.1f}mm), found {n_clusters} clusters, {n_noise} noise points")
        
        # If only 1 cluster found, try with smaller eps
        if n_clusters == 1 and n_noise == 0:
            print(f"  [DBSCAN] Only 1 cluster found, trying smaller eps...")
            for factor in [2.5, 2.0, 1.5, 1.2, 1.0]:
                eps_retry = avg_spacing * factor
                clustering = DBSCAN(
                    eps=eps_retry,
                    min_samples=min_samples,
                    metric='euclidean',
                    n_jobs=-1
                ).fit(points)
                
                labels = clustering.labels_
                n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                n_noise = list(labels).count(-1)
                
                if n_clusters > 1:
                    print(f"  [DBSCAN] Retry with eps={eps_retry:.1f}mm: found {n_clusters} clusters")
                    break
            
            # If still 1 cluster, fall back to curvature-based segmentation
            if n_clusters == 1:
                print(f"  [DBSCAN] Still 1 cluster, using curvature-based segmentation")
                return self._curvature_based_segmentation(points)
        
        # Filter out noise points (label=-1)
        unique_labels = set(labels)
        clusters = []

        for label in unique_labels:
            if label == -1:
                continue  # Skip noise

            cluster_mask = labels == label
            cluster_indices = np.where(cluster_mask)[0]

            # Accept smaller clusters for finer segmentation
            if len(cluster_indices) >= max(10, self.config['min_segment_size'] // 3):
                clusters.append(cluster_indices)
        
        print(f"  [DBSCAN] Returning {len(clusters)} clusters with >= {max(10, self.config['min_segment_size'] // 3)} points")
        
        return clusters
    
    def _curvature_based_segmentation(self, points: np.ndarray) -> List[np.ndarray]:
        """
        Segment points based on curvature when DBSCAN fails.
        
        Uses PCA to classify points as:
        - High curvature (edges, corners)
        - Medium curvature (blends, fillets)
        - Low curvature (flat surfaces, cylinders)
        """
        # Compute per-point curvatures
        curvatures = self._compute_point_curvatures(points)
        
        # Use percentile-based thresholds
        high_thresh = np.percentile(curvatures, 80)
        med_thresh = np.percentile(curvatures, 50)
        
        clusters = []
        
        # High curvature points
        high_mask = curvatures >= high_thresh
        if np.sum(high_mask) >= 10:
            clusters.append(np.where(high_mask)[0])
        
        # Medium curvature points
        med_mask = (curvatures >= med_thresh) & (curvatures < high_thresh)
        if np.sum(med_mask) >= 10:
            clusters.append(np.where(med_mask)[0])
        
        # Low curvature points
        low_mask = curvatures < med_thresh
        if np.sum(low_mask) >= 10:
            clusters.append(np.where(low_mask)[0])
        
        print(f"  [Curvature] Segmented into {len(clusters)} clusters based on curvature")
        
        return clusters
    
    def _fallback_hierarchical_segmentation(self, points: np.ndarray) -> List[SegmentResult]:
        """Fallback to original hierarchical segmentation if DBSCAN fails."""
        # Level 0: Coarse segmentation
        coarse_segments = self._segment_coarse(points)

        # Level 1: Medium segmentation (subdivide coarse segments)
        medium_segments = []
        for seg in coarse_segments:
            if len(seg.point_indices) > self.config['min_segment_size'] * 2:
                sub_segments = self._segment_medium(
                    points[seg.point_indices],
                    seg.point_indices,
                    seg.segment_id
                )
                medium_segments.extend(sub_segments)
                seg.children_ids = [s.segment_id for s in sub_segments]

        # Level 2: Fine segmentation (subdivide medium segments)
        fine_segments = []
        for seg in medium_segments:
            if len(seg.point_indices) > self.config['min_segment_size'] * 3:
                sub_segments = self._segment_fine(
                    points[seg.point_indices],
                    seg.point_indices,
                    seg.segment_id
                )
                fine_segments.extend(sub_segments)
                seg.children_ids = [s.segment_id for s in sub_segments]

        # Return only the finest level segments that exist
        if fine_segments:
            return fine_segments
        elif medium_segments:
            return medium_segments
        else:
            return coarse_segments
    
    def _segment_coarse(self, points: np.ndarray) -> List[SegmentResult]:
        """Level 0: Coarse segmentation by geometric primitives."""
        # Fit primitives to entire point cloud
        primitive_fit = self.primitive_fitter.fit_all_primitives(points)
        
        # Compute geometric properties
        geo_props = self._compute_geometric_properties(points)
        
        # Classify
        feature_type, confidence = self.classifier.classify(
            points, primitive_fit, geo_props
        )
        
        # Create segment
        segment = SegmentResult(
            segment_id='seg_0_0',
            feature_type=feature_type,
            point_indices=list(range(len(points))),
            confidence=confidence,
            geometric_params=primitive_fit,
            hierarchy_level=0
        )
        
        return [segment]
    
    def _segment_medium(self, points: np.ndarray, 
                       original_indices: List[int],
                       parent_id: str) -> List[SegmentResult]:
        """Level 1: Medium segmentation by curvature-based splitting."""
        # Compute per-point curvature
        curvatures = self._compute_point_curvatures(points)
        
        # Split by curvature threshold
        high_curvature_mask = curvatures > self.config['curvature_threshold']
        
        segments = []
        seg_counter = 0
        
        # High curvature segment (edges, corners)
        if np.sum(high_curvature_mask) > self.config['min_segment_size']:
            high_points = points[high_curvature_mask]
            high_indices = [original_indices[i] for i in np.where(high_curvature_mask)[0]]
            
            primitive_fit = self.primitive_fitter.fit_all_primitives(high_points)
            geo_props = self._compute_geometric_properties(high_points)
            feature_type, confidence = self.classifier.classify(
                high_points, primitive_fit, geo_props
            )
            
            segments.append(SegmentResult(
                segment_id=f'{parent_id}_1_{seg_counter}',
                feature_type=feature_type,
                point_indices=high_indices,
                confidence=confidence,
                geometric_params=primitive_fit,
                hierarchy_level=1,
                parent_id=parent_id
            ))
            seg_counter += 1
        
        # Low curvature segment (flat surfaces)
        low_curvature_mask = ~high_curvature_mask
        if np.sum(low_curvature_mask) > self.config['min_segment_size']:
            low_points = points[low_curvature_mask]
            low_indices = [original_indices[i] for i in np.where(low_curvature_mask)[0]]
            
            primitive_fit = self.primitive_fitter.fit_all_primitives(low_points)
            geo_props = self._compute_geometric_properties(low_points)
            feature_type, confidence = self.classifier.classify(
                low_points, primitive_fit, geo_props
            )
            
            segments.append(SegmentResult(
                segment_id=f'{parent_id}_1_{seg_counter}',
                feature_type=feature_type,
                point_indices=low_indices,
                confidence=confidence,
                geometric_params=primitive_fit,
                hierarchy_level=1,
                parent_id=parent_id
            ))
        
        return segments
    
    def _segment_fine(self, points: np.ndarray,
                     original_indices: List[int],
                     parent_id: str) -> List[SegmentResult]:
        """Level 2: Fine segmentation by spatial clustering."""
        from scipy.spatial import KDTree
        
        # Build KD-tree for spatial queries
        tree = KDTree(points)
        
        # Cluster points by proximity
        labels = self._spatial_clustering(points, tree)
        
        segments = []
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            mask = labels == label
            if np.sum(mask) < self.config['min_segment_size']:
                continue
            
            seg_points = points[mask]
            seg_indices = [original_indices[j] for j in np.where(mask)[0]]
            
            primitive_fit = self.primitive_fitter.fit_all_primitives(seg_points)
            geo_props = self._compute_geometric_properties(seg_points)
            feature_type, confidence = self.classifier.classify(
                seg_points, primitive_fit, geo_props
            )
            
            segments.append(SegmentResult(
                segment_id=f'{parent_id}_2_{i}',
                feature_type=feature_type,
                point_indices=seg_indices,
                confidence=confidence,
                geometric_params=primitive_fit,
                hierarchy_level=2,
                parent_id=parent_id
            ))
        
        return segments
    
    def _compute_point_curvatures(self, points: np.ndarray, k: int = 20) -> np.ndarray:
        """Compute per-point curvature using PCA."""
        from scipy.spatial import KDTree
        
        tree = KDTree(points)
        curvatures = np.zeros(len(points))
        
        for i, point in enumerate(points):
            distances, indices = tree.query(point, k=min(k, len(points)))
            neighbors = points[indices]
            
            if len(neighbors) < 3:
                continue
            
            # PCA
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov = np.cov(centered.T)
            
            if cov.shape == (3, 3):
                eigvals = np.linalg.eigvalsh(cov)
                # Curvature = ratio of smallest eigenvalue to sum
                curvatures[i] = eigvals[0] / (np.sum(eigvals) + 1e-10)
        
        return curvatures
    
    def _spatial_clustering(self, points: np.ndarray,
                           tree, radius: float = None) -> np.ndarray:
        """Simple spatial clustering using DBSCAN with adaptive eps."""
        from sklearn.cluster import DBSCAN

        # Estimate radius based on local point density using k-NN
        if radius is None:
            k = min(10, len(points) - 1)
            distances, _ = tree.query(points, k=k+1)
            avg_spacing = np.mean(distances[:, -1])
            # Use 2-3x average spacing for fine-level clustering
            radius = avg_spacing * 2.5

        # DBSCAN clustering
        min_samples = max(5, len(points) // 20)
        clustering = DBSCAN(eps=radius, min_samples=min_samples).fit(points)

        return clustering.labels_
    
    def _compute_geometric_properties(self, points: np.ndarray) -> Dict:
        """Compute geometric properties of point cloud segment."""
        if len(points) < 3:
            return {
                'mean_curvature': 0,
                'planarity': 0,
                'circularity': 0,
                'aspect_ratio': 1,
                'bounding_box': None
            }
        
        # Bounding box
        bbox_min = points.min(axis=0)
        bbox_max = points.max(axis=0)
        bbox_size = bbox_max - bbox_min
        
        # Aspect ratio
        sorted_sizes = np.sort(bbox_size)[::-1]
        aspect_ratio = sorted_sizes[1] / (sorted_sizes[0] + 1e-10)
        
        # Planarity (from PCA)
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.cov(centered.T)
        eigvals = np.linalg.eigvalsh(cov)
        planarity = (eigvals[1] - eigvals[0]) / (eigvals[2] + 1e-10)
        
        # Circularity (how close to circle/sphere)
        distances = np.linalg.norm(points - centroid, axis=1)
        radius_std = np.std(distances) / (np.mean(distances) + 1e-10)
        circularity = 1.0 - min(radius_std, 1.0)
        
        # Mean curvature
        curvatures = self._compute_point_curvatures(points)
        mean_curvature = float(np.mean(curvatures))
        
        return {
            'mean_curvature': mean_curvature,
            'planarity': float(planarity),
            'circularity': float(circularity),
            'aspect_ratio': float(aspect_ratio),
            'bounding_box': {
                'min': bbox_min.tolist(),
                'max': bbox_max.tolist(),
                'sizes': bbox_size.tolist()  # Add sizes for classification
            }
        }


class GeometricFeatureSegmentation:
    """
    Main segmentation class for manufacturing feature extraction.
    
    Combines:
    - RANSAC primitive fitting
    - Manufacturing feature classification
    - Hierarchical segmentation (3 levels)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'curvature_threshold': 0.1,
            'min_segment_size': 30
        }
        self.segmenter = HierarchicalSegmenter(self.config)
    
    def segment_features(self, points: np.ndarray) -> Dict:
        """
        Segment point cloud into manufacturing features.
        
        Args:
            points: Input point cloud (N, 3)
            
        Returns:
            Dictionary of segments with feature types and point indices
        """
        # Perform hierarchical segmentation
        segments = self.segmenter.segment(points)
        
        # Convert to output format
        result = {}
        for seg in segments:
            result[seg.segment_id] = {
                'feature_type': seg.feature_type,
                'nodes': seg.point_indices,  # For compatibility
                'point_indices': seg.point_indices,
                'confidence': seg.confidence,
                'hierarchy_level': seg.hierarchy_level,
                'parent_id': seg.parent_id,
                'children_ids': seg.children_ids,
                'geometric_params': seg.geometric_params
            }
        
        return result
