#!/usr/bin/env python3
"""
Feature segmentation module for point cloud manufacturing features.
Segments 3D point clouds into manufacturing features (holes, slots, pockets, etc.)
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
from typing import Dict, List, Tuple, Optional
import json


class FeatureSegmentation:
    """Segmentation of manufacturing features in point clouds."""
    
    def __init__(self, config=None):
        self.config = config or {
            'k_neighbors': 10,
            'edge_distance_threshold': 0.1,
            'dbscan_eps': 0.15,
            'dbscan_min_samples': 5,
            'curvature_threshold': 0.5,
            'normal_variance_threshold': 0.1
        }
        
        self.feature_types = [
            'hole', 'slot', 'pocket', 'chamfer', 'fillet',
            'step', 'island', 'counterbore', 'countersink', 'taper_hole',
            'plane', 'cylinder', 'cone', 'sphere', 'torus'
        ]
    
    def compute_normals(self, points: np.ndarray, k: int = None) -> np.ndarray:
        """
        Compute surface normals using PCA on local neighborhoods.
        
        Args:
            points: Nx3 array of point coordinates
            k: Number of neighbors for local fitting (default: 10)
            
        Returns:
            Nx3 array of normal vectors
        """
        k = k or self.config['k_neighbors']
        tree = KDTree(points)
        normals = np.zeros_like(points)
        
        for i, point in enumerate(points):
            distances, indices = tree.query(point, k=min(k, len(points)))
            neighbors = points[indices]
            
            # PCA for normal computation
            centroid = np.mean(neighbors, axis=0)
            centered = neighbors - centroid
            cov = np.cov(centered.T)
            
            if cov.shape == (3, 3):
                eigvals, eigvecs = np.linalg.eigh(cov)
                # Normal is eigenvector with smallest eigenvalue
                normal = eigvecs[:, 0]
                
                # Orient normals consistently (outward)
                if np.dot(normal, centroid) < 0:
                    normal = -normal
                    
                normals[i] = normal
            else:
                normals[i] = [0, 0, 1]  # Default normal
        
        return normals
    
    def compute_curvature(self, points: np.ndarray, normals: np.ndarray, 
                         k: int = None) -> np.ndarray:
        """
        Compute surface curvature as variation in normal directions.
        
        Args:
            points: Nx3 array of point coordinates
            normals: Nx3 array of normal vectors
            k: Number of neighbors (default: 10)
            
        Returns:
            N array of curvature values
        """
        k = k or self.config['k_neighbors']
        tree = KDTree(points)
        curvatures = np.zeros(len(points))
        
        for i, point in enumerate(points):
            distances, indices = tree.query(point, k=min(k, len(points)))
            neighbor_normals = normals[indices]
            
            # Curvature as mean angular difference
            normal_diff = np.arccos(np.clip(
                np.dot(neighbor_normals, normals[i]), -1, 1
            ))
            curvatures[i] = np.mean(normal_diff)
        
        return curvatures
    
    def compute_feature_descriptors(self, points: np.ndarray, 
                                   normals: np.ndarray,
                                   curvatures: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Compute additional feature descriptors for classification.
        
        Returns:
            Dictionary of feature descriptors
        """
        tree = KDTree(points)
        k = self.config['k_neighbors']
        
        descriptors = {
            'point_gaussian_curvature': curvatures,
            'point_mean_curvature': curvatures,  # Simplified
            'normal_variation': np.zeros(len(points)),
            'local_density': np.zeros(len(points)),
            'planarity': np.zeros(len(points)),
            'linearity': np.zeros(len(points)),
            'sphericity': np.zeros(len(points))
        }
        
        for i, point in enumerate(points):
            distances, indices = tree.query(point, k=min(k, len(points)))
            neighbors = points[indices]
            neighbor_normals = normals[indices]
            
            # Normal variation
            descriptors['normal_variation'][i] = np.var(neighbor_normals, axis=0).sum()
            
            # Local density
            if distances[-1] > 0:
                descriptors['local_density'][i] = len(indices) / (4/3 * np.pi * distances[-1]**3)
            
            # Shape features from covariance
            centered = neighbors - np.mean(neighbors, axis=0)
            cov = np.cov(centered.T)
            if cov.shape == (3, 3):
                eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]
                sum_eig = eigvals.sum() + 1e-10
                
                descriptors['planarity'][i] = (eigvals[0] - eigvals[1]) / sum_eig
                descriptors['linearity'][i] = (eigvals[1] - eigvals[2]) / sum_eig
                descriptors['sphericity'][i] = eigvals[2] / sum_eig
        
        return descriptors
    
    def build_graph_from_pointcloud(self, points: np.ndarray, 
                                    normals: np.ndarray,
                                    curvatures: np.ndarray,
                                    descriptors: Dict[str, np.ndarray] = None
                                    ) -> nx.Graph:
        """
        Build a graph from point cloud with geometric attributes.
        
        Args:
            points: Nx3 array of point coordinates
            normals: Nx3 array of normal vectors
            curvatures: N array of curvature values
            descriptors: Optional dictionary of additional descriptors
            
        Returns:
            NetworkX graph with nodes and edges
        """
        G = nx.Graph()
        tree = KDTree(points)
        
        # Add nodes with attributes
        for i, (point, normal, curvature) in enumerate(zip(points, normals, curvatures)):
            node_attrs = {
                'position': point.tolist(),
                'normal': normal.tolist(),
                'curvature': float(curvature),
                'point_index': i
            }
            
            if descriptors:
                for key, values in descriptors.items():
                    node_attrs[key] = float(values[i])
            
            G.add_node(i, **node_attrs)
        
        # Add edges based on spatial proximity
        distances, indices = tree.query(points, k=10)
        
        for i in range(len(points)):
            for j, dist in zip(indices[i], distances[i]):
                if i != j and dist < self.config['edge_distance_threshold']:
                    G.add_edge(i, j, distance=float(dist))
        
        return G
    
    def segment_graph(self, G: nx.Graph) -> Dict[int, List[int]]:
        """
        Segment graph into communities using Louvain algorithm.
        
        Args:
            G: NetworkX graph
            
        Returns:
            Dictionary mapping segment IDs to node lists
        """
        try:
            from community import community_louvain
            partition = community_louvain.best_partition(G, resolution=1.0)
        except ImportError:
            # Fallback to DBSCAN if community detection not available
            positions = np.array([G.nodes[n]['position'] for n in G.nodes()])
            db = DBSCAN(eps=self.config['dbscan_eps'], 
                       min_samples=self.config['dbscan_min_samples'])
            labels = db.fit_predict(positions)
            partition = {n: int(labels[n]) for n in G.nodes()}
        
        # Convert to segments dictionary
        segments = {}
        for node, community_id in partition.items():
            if community_id not in segments:
                segments[community_id] = []
            segments[community_id].append(node)
        
        return segments
    
    def classify_segment(self, G: nx.Graph, nodes: List[int],
                        descriptors: Dict[str, np.ndarray] = None) -> Dict:
        """
        Classify a segment as a manufacturing feature.
        
        Args:
            G: NetworkX graph
            nodes: List of node indices in segment
            descriptors: Optional precomputed descriptors
            
        Returns:
            Classification result dictionary
        """
        normals = np.array([G.nodes[n]['normal'] for n in nodes])
        curvatures = np.array([G.nodes[n]['curvature'] for n in nodes])
        positions = np.array([G.nodes[n]['position'] for n in nodes])
        
        # Compute shape descriptors
        normal_variance = np.var(normals, axis=0)
        mean_curvature = np.mean(curvatures)
        max_curvature = np.max(curvatures)
        
        # Geometric features
        centroid = np.mean(positions, axis=0)
        bbox = np.max(positions, axis=0) - np.min(positions, axis=0)
        aspect_ratio = np.max(bbox) / (np.min(bbox[bbox > 0]) + 1e-10)
        
        # Classification rules
        feature_type = 'unknown'
        confidence = 0.5
        
        # Hole detection: high curvature, cylindrical pattern
        if mean_curvature > self.config['curvature_threshold']:
            # Check for cylindrical pattern
            if aspect_ratio > 2:  # Elongated
                feature_type = 'taper_hole'
                confidence = 0.7
            else:
                feature_type = 'hole'
                confidence = 0.8
        
        # Pocket detection: low curvature, flat with boundaries
        elif np.all(normal_variance < self.config['normal_variance_threshold']):
            if aspect_ratio > 3:
                feature_type = 'slot'
                confidence = 0.75
            else:
                feature_type = 'pocket'
                confidence = 0.7
        
        # Chamfer/fillet detection: variable normal orientation
        elif np.any(normal_variance > 0.5):
            if mean_curvature > 0.3:
                feature_type = 'fillet'
                confidence = 0.6
            else:
                feature_type = 'chamfer'
                confidence = 0.65
        
        # Step detection
        elif len(nodes) > 100 and aspect_ratio > 1.5:
            feature_type = 'step'
            confidence = 0.6
        
        # Plane detection
        elif mean_curvature < 0.1 and np.all(normal_variance < 0.05):
            feature_type = 'plane'
            confidence = 0.9
        
        # Cylinder detection
        elif 0.2 < mean_curvature < 0.5:
            feature_type = 'cylinder'
            confidence = 0.75
        
        return {
            'feature_type': feature_type,
            'confidence': confidence,
            'geometric_properties': {
                'centroid': centroid.tolist(),
                'bbox_size': bbox.tolist(),
                'aspect_ratio': float(aspect_ratio),
                'mean_curvature': float(mean_curvature),
                'normal_variance': normal_variance.tolist(),
                'num_points': len(nodes)
            }
        }
    
    def segment_features(self, points: np.ndarray, 
                        descriptors: Dict[str, np.ndarray] = None) -> Dict:
        """
        Complete feature segmentation pipeline.
        
        Args:
            points: Nx3 array of point coordinates
            descriptors: Optional precomputed descriptors
            
        Returns:
            Dictionary of segmented features
        """
        # Compute geometric properties
        normals = self.compute_normals(points)
        curvatures = self.compute_curvature(points, normals)
        
        if descriptors is None:
            descriptors = self.compute_feature_descriptors(points, normals, curvatures)
        
        # Build graph
        G = self.build_graph_from_pointcloud(points, normals, curvatures, descriptors)
        
        # Segment graph
        segments = self.segment_graph(G)
        
        # Classify each segment
        feature_segments = {}
        for seg_id, nodes in segments.items():
            if len(nodes) < 5:  # Skip very small segments
                continue
                
            classification = self.classify_segment(G, nodes, descriptors)
            
            feature_segments[seg_id] = {
                'nodes': nodes,
                'feature_type': classification['feature_type'],
                'confidence': classification['confidence'],
                'geometric_properties': classification['geometric_properties'],
                'point_indices': nodes
            }
        
        return feature_segments
    
    def compute_feature_dimensions(self, points: np.ndarray,
                                   feature_segments: Dict) -> Dict:
        """
        Compute dimensional measurements for each feature segment.
        
        Args:
            points: Nx3 array of point coordinates
            feature_segments: Dictionary of segmented features
            
        Returns:
            Dictionary with dimensional measurements
        """
        dimensions = {}
        
        for seg_id, segment in feature_segments.items():
            nodes = segment['nodes']
            feature_points = points[nodes]
            feature_type = segment['feature_type']
            
            dim = {
                'feature_type': feature_type,
                'segment_id': seg_id
            }
            
            if feature_type in ['hole', 'cylinder', 'taper_hole']:
                # Fit cylinder and get diameter
                dim.update(self._fit_cylinder_dimensions(feature_points))
            elif feature_type in ['slot', 'pocket']:
                # Get bounding box dimensions
                dim.update(self._fit_bbox_dimensions(feature_points))
            elif feature_type == 'plane':
                # Get plane dimensions
                dim.update(self._fit_plane_dimensions(feature_points))
            elif feature_type == 'sphere':
                # Fit sphere
                dim.update(self._fit_sphere_dimensions(feature_points))
            else:
                # Generic bounding box
                dim.update(self._fit_bbox_dimensions(feature_points))
            
            dimensions[seg_id] = dim
        
        return dimensions
    
    def _fit_cylinder_dimensions(self, points: np.ndarray) -> Dict:
        """Fit cylinder to points and return dimensions."""
        # Simple cylinder fitting using PCA
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Axis direction is eigenvector with largest eigenvalue
        axis = eigvecs[:, 2]
        
        # Project points to plane perpendicular to axis
        proj_matrix = np.eye(3) - np.outer(axis, axis)
        projected = centered @ proj_matrix.T
        
        # Radius from mean distance to axis
        distances = np.linalg.norm(projected, axis=1)
        radius = np.mean(distances)
        std_radius = np.std(distances)
        
        # Height from extent along axis
        heights = centered @ axis
        height = np.max(heights) - np.min(heights)
        
        return {
            'diameter': float(2 * radius),
            'diameter_std': float(std_radius),
            'radius': float(radius),
            'height': float(height),
            'axis_direction': axis.tolist(),
            'centroid': centroid.tolist()
        }
    
    def _fit_bbox_dimensions(self, points: np.ndarray) -> Dict:
        """Compute bounding box dimensions."""
        bbox_min = np.min(points, axis=0)
        bbox_max = np.max(points, axis=0)
        dimensions = bbox_max - bbox_min
        
        return {
            'bbox_min': bbox_min.tolist(),
            'bbox_max': bbox_max.tolist(),
            'dimensions': dimensions.tolist(),
            'length': float(dimensions[0]),
            'width': float(dimensions[1]),
            'height': float(dimensions[2]),
            'centroid': np.mean(points, axis=0).tolist()
        }
    
    def _fit_plane_dimensions(self, points: np.ndarray) -> Dict:
        """Fit plane and compute in-plane dimensions."""
        centroid = np.mean(points, axis=0)
        centered = points - centroid
        cov = np.cov(centered.T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        
        # Normal is eigenvector with smallest eigenvalue
        normal = eigvecs[:, 0]
        
        # In-plane dimensions from other eigenvectors
        in_plane_dims = np.sort(eigvals[1:])[::-1]
        
        return {
            'normal': normal.tolist(),
            'primary_dimension': float(2 * np.sqrt(in_plane_dims[0])),
            'secondary_dimension': float(2 * np.sqrt(in_plane_dims[1])),
            'planarity': float(1 - eigvals[0] / (eigvals.sum() + 1e-10)),
            'centroid': centroid.tolist()
        }
    
    def _fit_sphere_dimensions(self, points: np.ndarray) -> Dict:
        """Fit sphere to points."""
        centroid = np.mean(points, axis=0)
        distances = np.linalg.norm(points - centroid, axis=1)
        radius = np.mean(distances)
        std_radius = np.std(distances)
        
        return {
            'diameter': float(2 * radius),
            'diameter_std': float(std_radius),
            'radius': float(radius),
            'centroid': centroid.tolist(),
            'sphericity': float(1 - std_radius / (radius + 1e-10))
        }


def save_feature_segments(segments: Dict, output_file: str):
    """Save feature segments to JSON file."""
    with open(output_file, 'w') as f:
        json.dump(segments, f, indent=2)


def load_feature_segments(input_file: str) -> Dict:
    """Load feature segments from JSON file."""
    with open(input_file, 'r') as f:
        return json.load(f)
