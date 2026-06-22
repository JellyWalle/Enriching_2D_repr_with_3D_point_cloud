#!/usr/bin/env python3
"""
Feature segmentation script for manufacturing features in point clouds.
Segments 3D point clouds into manufacturing features (holes, slots, pockets, etc.)
and saves segmentation results to JSON files.
"""

import numpy as np
import networkx as nx
from scipy.spatial import KDTree
from sklearn.cluster import DBSCAN
import os
import json
import glob
from tqdm import tqdm
from typing import Dict, List, Optional
import argparse

# Import from local module
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_segmentation import FeatureSegmentation, save_feature_segments


class PointCloudFeatureSegmenter:
    """
    High-level interface for segmenting manufacturing features in point clouds.
    Combines FeatureSegmentation with additional processing and visualization.
    """
    
    def __init__(self, config=None):
        self.config = config or {
            'k_neighbors': 10,
            'edge_distance_threshold': 0.1,
            'dbscan_eps': 0.15,
            'dbscan_min_samples': 5,
            'curvature_threshold': 0.5,
            'normal_variance_threshold': 0.1,
            'min_segment_size': 10,
            'max_segment_size': 10000
        }
        
        self.segmenter = FeatureSegmentation(self.config)
    
    def segment_pointcloud(self, points: np.ndarray, 
                          save_visualization: bool = False,
                          output_dir: str = None) -> Dict:
        """
        Complete segmentation pipeline for a point cloud.
        
        Args:
            points: Nx3 array of point coordinates
            save_visualization: Whether to save visualization
            output_dir: Directory for output files
            
        Returns:
            Dictionary with segmentation results
        """
        # Compute geometric properties
        print("Computing normals and curvatures...")
        normals = self.segmenter.compute_normals(points)
        curvatures = self.segmenter.compute_curvature(points, normals)
        
        # Compute feature descriptors
        print("Computing feature descriptors...")
        descriptors = self.segmenter.compute_feature_descriptors(
            points, normals, curvatures
        )
        
        # Build graph
        print("Building adjacency graph...")
        G = self.segmenter.build_graph_from_pointcloud(
            points, normals, curvatures, descriptors
        )
        
        # Segment graph
        print("Segmenting graph into communities...")
        segments = self.segmenter.segment_graph(G)
        
        # Filter small segments
        filtered_segments = {
            seg_id: nodes for seg_id, nodes in segments.items()
            if self.config['min_segment_size'] <= len(nodes) <= self.config['max_segment_size']
        }
        
        # Classify segments
        print("Classifying segments...")
        feature_segments = {}
        for seg_id, nodes in filtered_segments.items():
            classification = self.segmenter.classify_segment(G, nodes, descriptors)
            
            feature_segments[seg_id] = {
                'nodes': nodes,
                'feature_type': classification['feature_type'],
                'confidence': classification['confidence'],
                'geometric_properties': classification['geometric_properties'],
                'point_indices': nodes,
                'descriptors': {
                    'mean_curvature': float(np.mean(curvatures[nodes])),
                    'normal_variation': float(np.mean(descriptors['normal_variation'][nodes])),
                    'planarity': float(np.mean(descriptors['planarity'][nodes])),
                    'linearity': float(np.mean(descriptors['linearity'][nodes]))
                }
            }
        
        # Compute dimensions for each feature
        print("Computing feature dimensions...")
        dimensions = self.segmenter.compute_feature_dimensions(points, feature_segments)
        
        for seg_id in feature_segments:
            if seg_id in dimensions:
                feature_segments[seg_id]['dimensions'] = dimensions[seg_id]
        
        # Create result summary
        result = {
            'num_points': len(points),
            'num_segments': len(feature_segments),
            'feature_types': self._count_feature_types(feature_segments),
            'segments': feature_segments,
            'config': self.config
        }
        
        # Save visualization if requested
        if save_visualization and output_dir:
            self._save_segmentation_visualization(
                points, feature_segments, output_dir
            )
        
        return result
    
    def _count_feature_types(self, feature_segments: Dict) -> Dict[str, int]:
        """Count occurrences of each feature type."""
        type_counts = {}
        for seg_data in feature_segments.values():
            ftype = seg_data['feature_type']
            type_counts[ftype] = type_counts.get(ftype, 0) + 1
        return type_counts
    
    def _save_segmentation_visualization(self, points: np.ndarray,
                                         feature_segments: Dict,
                                         output_dir: str):
        """Save visualization of segmentation results."""
        try:
            import open3d as o3d
            
            # Create color map for different feature types
            feature_types = list(set(
                seg['feature_type'] for seg in feature_segments.values()
            ))
            color_map = self._generate_color_map(feature_types)
            
            # Create colored point cloud
            colors = np.zeros_like(points)
            
            for seg_id, seg_data in feature_segments.items():
                nodes = seg_data['nodes']
                ftype = seg_data['feature_type']
                color = color_map.get(ftype, [1, 1, 1])
                colors[nodes] = color
            
            # Create Open3D point cloud
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save visualization
            output_file = os.path.join(output_dir, 'segmented_cloud.ply')
            o3d.io.write_point_cloud(output_file, pcd)
            print(f"Segmentation visualization saved to {output_file}")
            
        except ImportError:
            print("Open3D not available, skipping visualization")
        except Exception as e:
            print(f"Error saving visualization: {e}")
    
    def _generate_color_map(self, feature_types: List[str]) -> Dict[str, List[float]]:
        """Generate color map for feature types."""
        # Predefined colors for common feature types
        predefined_colors = {
            'hole': [1, 0, 0],        # Red
            'slot': [0, 1, 0],        # Green
            'pocket': [0, 0, 1],      # Blue
            'chamfer': [1, 1, 0],     # Yellow
            'fillet': [1, 0, 1],      # Magenta
            'step': [0, 1, 1],        # Cyan
            'plane': [0.5, 0.5, 0.5], # Gray
            'cylinder': [1, 0.5, 0],  # Orange
            'cone': [0.5, 0, 1],      # Purple
            'sphere': [0, 0.5, 0.5],  # Teal
            'unknown': [0.5, 0.5, 0.5]  # Gray
        }
        
        color_map = {}
        available_colors = [
            [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1],
            [0.5, 0.5, 0.5], [1, 0.5, 0], [0.5, 0, 1], [0, 0.5, 0.5],
            [0.5, 0, 0], [0, 0.5, 0], [0, 0, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]
        ]
        
        color_idx = 0
        for ftype in feature_types:
            if ftype in predefined_colors:
                color_map[ftype] = predefined_colors[ftype]
            else:
                color_map[ftype] = available_colors[color_idx % len(available_colors)]
                color_idx += 1
        
        return color_map
    
    def process_directory(self, input_dir: str, output_dir: str,
                         pattern: str = '*_none.npy',
                         max_files: int = None) -> Dict:
        """
        Process all point cloud files in a directory.
        
        Args:
            input_dir: Directory with point cloud .npy files
            output_dir: Directory for output files
            pattern: Glob pattern for input files
            max_files: Maximum number of files to process
            
        Returns:
            Summary statistics
        """
        # Find all matching files
        pointcloud_files = glob.glob(os.path.join(input_dir, pattern))
        
        if max_files:
            pointcloud_files = pointcloud_files[:max_files]
        
        print(f"Found {len(pointcloud_files)} point cloud files to process")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Process files
        successful = 0
        failed = 0
        total_features = 0
        
        for pc_file in tqdm(pointcloud_files, desc="Segmenting features"):
            try:
                # Load point cloud
                points = np.load(pc_file)
                
                # Segment
                result = self.segment_pointcloud(points)
                
                # Save results
                base_name = os.path.splitext(os.path.basename(pc_file))[0]
                output_file = os.path.join(output_dir, f"{base_name}_segments.json")
                save_feature_segments(result, output_file)
                
                successful += 1
                total_features += result['num_segments']
                
            except Exception as e:
                print(f"\nError processing {pc_file}: {e}")
                failed += 1
        
        # Save summary
        summary = {
            'total_files': len(pointcloud_files),
            'successful': successful,
            'failed': failed,
            'success_rate': successful / len(pointcloud_files) * 100 if pointcloud_files else 0,
            'total_features_segmented': total_features,
            'avg_features_per_cloud': total_features / successful if successful > 0 else 0
        }
        
        summary_file = os.path.join(output_dir, 'segmentation_summary.json')
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSegmentation complete!")
        print(f"  Successful: {successful}")
        print(f"  Failed: {failed}")
        print(f"  Total features: {total_features}")
        print(f"  Summary saved to: {summary_file}")
        
        return summary


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description='Segment manufacturing features in point clouds'
    )
    parser.add_argument('-i', '--input', dest='input_dir', required=True,
                        help='Input directory with point cloud .npy files')
    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                        help='Output directory for segmentation results')
    parser.add_argument('--pattern', default='*_none.npy',
                        help='Glob pattern for input files')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process')
    parser.add_argument('--visualize', action='store_true',
                        help='Save visualization of segmentation')
    parser.add_argument('--k-neighbors', type=int, default=10,
                        help='Number of neighbors for normal/curvature computation')
    parser.add_argument('--edge-threshold', type=float, default=0.1,
                        help='Edge distance threshold for graph building')
    parser.add_argument('--curvature-threshold', type=float, default=0.5,
                        help='Curvature threshold for feature classification')
    
    args = parser.parse_args()
    
    # Create configuration
    config = {
        'k_neighbors': args.k_neighbors,
        'edge_distance_threshold': args.edge_threshold,
        'curvature_threshold': args.curvature_threshold,
        'dbscan_eps': args.edge_threshold * 1.5,
        'dbscan_min_samples': 5,
        'normal_variance_threshold': 0.1,
        'min_segment_size': 10,
        'max_segment_size': 10000
    }
    
    # Create segmenter
    segmenter = PointCloudFeatureSegmenter(config)
    
    # Process directory
    segmenter.process_directory(
        args.input_dir,
        args.output_dir,
        pattern=args.pattern,
        max_files=args.max_files
    )


if __name__ == '__main__':
    main()
