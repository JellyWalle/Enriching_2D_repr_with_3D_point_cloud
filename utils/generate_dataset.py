#!/usr/bin/env python3
"""
Script to generate complete dataset from ABC dataset STEP files
With feature correspondences between 2D drawings and 3D point clouds
"""

import os
import sys
import json
from pathlib import Path
from tqdm import tqdm
import argparse
import numpy as np

# Add parent directory to path to import render_utils
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from render_with_points_utils import (
    process_step_file_with_tolerances_and_pointcloud,
    read_step_file,
    FeatureCorrespondenceCreator,
    UnifiedScaler,
    ProjectionMatrix3DTo2D,
    create_feature_correspondences
)


def generate_dataset(input_dir, output_dir, max_files=None, start_idx=0,
                    generate_correspondences=True, generate_segmentation=True):
    """
    Generate dataset from ABC dataset STEP files with feature correspondences.

    Args:
        input_dir: Directory containing STEP files
        output_dir: Output directory for generated dataset
        max_files: Maximum number of files to process (None for all)
        start_idx: Starting index for processing
        generate_correspondences: Whether to generate feature correspondences
        generate_segmentation: Whether to generate feature segmentation
    """

    # Create output directories
    drawings_dir = os.path.join(output_dir, 'drawings')
    pointclouds_dir = os.path.join(output_dir, 'pointclouds')
    annotations_dir = os.path.join(output_dir, 'annotations')
    correspondences_dir = os.path.join(output_dir, 'correspondences')
    segments_dir = os.path.join(output_dir, 'segments')

    for d in [drawings_dir, pointclouds_dir, annotations_dir, 
              correspondences_dir, segments_dir]:
        os.makedirs(d, exist_ok=True)

    # Find all STEP files
    step_files = []
    for root, dirs, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith(('.step', '.stp')):
                step_files.append(os.path.join(root, file))

    if max_files is not None:
        step_files = step_files[start_idx:start_idx + max_files]
    else:
        step_files = step_files[start_idx:]

    print(f"Found {len(step_files)} STEP files to process")
    print(f"Output directories:")
    print(f"  Drawings: {drawings_dir}")
    print(f"  Point clouds: {pointclouds_dir}")
    print(f"  Annotations: {annotations_dir}")
    if generate_correspondences:
        print(f"  Correspondences: {correspondences_dir}")
    if generate_segmentation:
        print(f"  Segments: {segments_dir}")

    # Initialize correspondence creator
    if generate_correspondences:
        correspondence_creator = FeatureCorrespondenceCreator()
    
    # Initialize scaler for unified normalization
    unified_scaler = UnifiedScaler(target_scale=1.0, normalize_center=True)

    # Process files
    successful = 0
    failed = 0
    total_correspondences = 0
    total_features = 0

    for step_file in tqdm(step_files, desc="Processing STEP files"):
        try:
            base_name = os.path.splitext(os.path.basename(step_file))[0]
            
            # Load shape
            shape = read_step_file(step_file)
            
            # Generate base annotation (drawing + point cloud)
            annotation = process_step_file_with_tolerances_and_pointcloud(
                step_file, output_dir,
                generate_pointcloud=True,
                generate_tolerances=True
            )
            
            # Load generated point cloud
            pc_file = os.path.join(output_dir, f"{base_name}_pc_ideal.npy")
            if os.path.exists(pc_file):
                points_3d = np.load(pc_file)
                
                # Fit unified scaler on first file
                if successful == 0:
                    unified_scaler.fit(points_3d)
                
                # Transform points to normalized coordinates
                points_normalized = unified_scaler.transform_points(points_3d)
                
                # Save normalized point cloud
                pc_normalized_file = os.path.join(pointclouds_dir, f"{base_name}_normalized.npy")
                np.save(pc_normalized_file, points_normalized)
                
                # Generate feature correspondences
                if generate_correspondences:
                    # Create dummy drawing dict (in real implementation, this would come from the drawing generator)
                    drawing_dict = {'front': {'dimensions': []}}
                    
                    correspondences = create_feature_correspondences(
                        shape, drawing_dict, points_normalized, view_type='front'
                    )
                    
                    # Save correspondences
                    corr_file = os.path.join(correspondences_dir, f"{base_name}_correspondences.json")
                    with open(corr_file, 'w') as f:
                        json.dump({
                            'step_file': step_file,
                            'point_cloud': pc_normalized_file,
                            'view_type': 'front',
                            'correspondences': correspondences,
                            'normalization_params': unified_scaler.get_normalization_params()
                        }, f, indent=2)
                    
                    total_correspondences += len(correspondences)
                    total_features += len(correspondences)
            
            successful += 1
            
        except Exception as e:
            print(f"\nError processing {step_file}: {e}")
            failed += 1

    # Generate dataset statistics
    stats = {
        'total_files': len(step_files),
        'successful': successful,
        'failed': failed,
        'success_rate': successful / len(step_files) * 100 if len(step_files) > 0 else 0,
        'total_correspondences': total_correspondences,
        'total_features': total_features,
        'avg_correspondences_per_file': total_correspondences / successful if successful > 0 else 0,
        'output_directories': {
            'drawings': drawings_dir,
            'pointclouds': pointclouds_dir,
            'annotations': annotations_dir,
            'correspondences': correspondences_dir if generate_correspondences else None,
            'segments': segments_dir if generate_segmentation else None
        },
        'normalization_params': unified_scaler.get_normalization_params()
    }

    stats_file = os.path.join(output_dir, 'dataset_stats.json')
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)

    print(f"\nDataset generation complete!")
    print(f"  Successful: {successful}")
    print(f"  Failed: {failed}")
    print(f"  Success rate: {stats['success_rate']:.2f}%")
    print(f"  Total correspondences: {total_correspondences}")
    print(f"  Statistics saved to: {stats_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Generate dataset from ABC dataset STEP files with feature correspondences'
    )
    parser.add_argument('-i', '--input', dest='input_dir', required=True,
                        help='Input directory with ABC dataset STEP files')
    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                        help='Output directory for generated dataset')
    parser.add_argument('--max-files', type=int, default=None,
                        help='Maximum number of files to process')
    parser.add_argument('--start-idx', type=int, default=0,
                        help='Starting index for processing')
    parser.add_argument('--no-correspondences', dest='generate_correspondences',
                        action='store_false', help='Do not generate feature correspondences')
    parser.add_argument('--no-segmentation', dest='generate_segmentation',
                        action='store_false', help='Do not generate feature segmentation')

    args = parser.parse_args()

    generate_dataset(
        args.input_dir,
        args.output_dir,
        max_files=args.max_files,
        start_idx=args.start_idx,
        generate_correspondences=args.generate_correspondences,
        generate_segmentation=args.generate_segmentation
    )


if __name__ == '__main__':
    main()