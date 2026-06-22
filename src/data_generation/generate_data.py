#!/usr/bin/env python3
"""
Data Generation Module for Hybrid GNN Dataset.

Generates 2D drawings and 3D point clouds from ABC dataset STEP files.
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

# Add utils to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))

from render_with_points_utils import (
    process_step_file_with_tolerances_and_pointcloud,
    read_step_file,
    PointCloudGenerator,
    ToleranceGenerator,
    FeatureCorrespondenceCreator,
    UnifiedScaler
)


class DatasetGenerator:
    """
    Generate complete dataset from ABC STEP files.
    
    Generates:
    - 2D engineering drawings with tolerances (front, top, side views)
    - 3D point clouds with controlled deviations
    - Correspondence annotations
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'num_points': 1024,
            'generate_views': ['front', 'top', 'side'],
            'deviation_types': ['ideal', 'in_tolerance', 'out_of_tolerance'],
            'tolerance_range': (0.01, 0.05),  # 1-5% of nominal
            'noise_std': 0.05,  # mm
        }
        
        self.stats = {
            'total_models': 0,
            'successful': 0,
            'failed': 0,
            'drawings_generated': 0,
            'pointclouds_generated': 0,
            'annotations_generated': 0
        }
    
    def generate_dataset(self, abc_dataset_path: str,
                        output_dir: str,
                        max_files: int = 150) -> Dict:
        """
        Generate complete dataset from ABC STEP files.
        
        Args:
            abc_dataset_path: Path to ABC dataset STEP files
            output_dir: Output directory for generated dataset
            max_files: Maximum number of files to process (default: 150)
            
        Returns:
            Dataset statistics
        """
        abc_path = Path(abc_dataset_path)
        output_path = Path(output_dir)
        
        # Create output directories
        drawings_dir = output_path / 'drawings'
        pointclouds_dir = output_path / 'pointclouds'
        annotations_dir = output_path / 'annotations'
        
        for d in [drawings_dir, pointclouds_dir, annotations_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Find STEP files
        step_files = list(abc_path.glob('**/*.step'))[:max_files]
        self.stats['total_models'] = len(step_files)
        
        print(f"Found {len(step_files)} STEP files to process")
        
        # Process each file
        for step_file in tqdm(step_files, desc="Generating dataset"):
            try:
                self._process_model(step_file, output_path)
                self.stats['successful'] += 1
            except Exception as e:
                print(f"\nError processing {step_file.name}: {e}")
                self.stats['failed'] += 1
        
        # Save statistics
        self._save_statistics(output_path / 'dataset_stats.json')
        
        return self.stats
    
    def _process_model(self, step_file: Path, output_path: Path):
        """Process single model: generate drawings, point clouds, annotations."""
        base_name = step_file.stem
        model_dir = output_path
        
        # 1. Generate 2D drawings (3 views)
        drawings = self._generate_drawings(step_file, model_dir)
        self.stats['drawings_generated'] += len(drawings)
        
        # 2. Generate 3D point clouds (3 deviation types)
        pointclouds = self._generate_pointclouds(step_file, model_dir)
        self.stats['pointclouds_generated'] += len(pointclouds)
        
        # 3. Generate annotations
        annotation = self._generate_annotation(
            step_file, drawings, pointclouds, model_dir
        )
        self.stats['annotations_generated'] += 1
    
    def _generate_drawings(self, step_file: Path,
                          output_path: Path) -> List[str]:
        """Generate 2D drawings for all views."""
        from render_with_points_utils import export_shape_to_svg_with_tolerances
        
        shape = read_step_file(str(step_file))
        drawings = []
        
        # View configurations
        views = {
            'front': {'location': (0, 0, 0), 'direction': (0, 0, 1)},
            'top': {'location': (0, 0, 0), 'direction': (0, 1, 0)},
            'side': {'location': (0, 0, 0), 'direction': (1, 0, 0)}
        }
        
        for view_name, config in views.items():
            svg_file = output_path / 'drawings' / f"{step_file.stem}_{view_name}.svg"
            
            try:
                export_shape_to_svg_with_tolerances(
                    shape,
                    filename=str(svg_file),
                    location=config['location'],
                    direction=config['direction'],
                    add_tolerances=True
                )
                drawings.append(str(svg_file))
            except Exception as e:
                print(f"  Warning: Could not generate {view_name} view: {e}")
        
        return drawings
    
    def _generate_pointclouds(self, step_file: Path,
                             output_path: Path) -> List[str]:
        """Generate 3D point clouds with different deviation types."""
        from render_with_points_utils import PointCloudGenerator
        
        pc_gen = PointCloudGenerator({
            'num_points': self.config['num_points'],
            'gaussian_noise_std': self.config['noise_std']
        })
        
        pointclouds = []
        base_name = step_file.stem
        
        # Generate ideal point cloud
        pc_ideal = output_path / 'pointclouds' / f"{base_name}_pc_ideal.npy"
        try:
            points = pc_gen.generate_pointcloud(str(step_file), str(pc_ideal))
            pointclouds.append(str(pc_ideal))
        except Exception as e:
            print(f"  Warning: Could not generate ideal point cloud: {e}")
        
        # Generate in-tolerance point cloud
        pc_in_tol = output_path / 'pointclouds' / f"{base_name}_pc_in_tol.npy"
        try:
            points = pc_gen.generate_pointcloud(
                str(step_file), str(pc_in_tol),
                deviation_type='in_tolerance',
                max_deviation=0.05
            )
            pointclouds.append(str(pc_in_tol))
        except Exception as e:
            print(f"  Warning: Could not generate in-tolerance point cloud: {e}")
        
        # Generate out-of-tolerance point cloud
        pc_out_tol = output_path / 'pointclouds' / f"{base_name}_pc_out_tol.npy"
        try:
            points = pc_gen.generate_pointcloud(
                str(step_file), str(pc_out_tol),
                deviation_type='out_of_tolerance',
                max_deviation=0.15
            )
            pointclouds.append(str(pc_out_tol))
        except Exception as e:
            print(f"  Warning: Could not generate out-of-tolerance point cloud: {e}")
        
        return pointclouds
    
    def _generate_annotation(self, step_file: Path,
                            drawings: List[str],
                            pointclouds: List[str],
                            output_path: Path) -> str:
        """Generate annotation file with correspondences."""
        base_name = step_file.stem
        
        annotation = {
            'model_id': base_name,
            'step_file': str(step_file),
            'drawings': drawings,
            'pointclouds': pointclouds,
            'metadata': {
                'num_drawings': len(drawings),
                'num_pointclouds': len(pointclouds),
                'views': ['front', 'top', 'side'],
                'deviation_types': ['ideal', 'in_tolerance', 'out_of_tolerance']
            }
        }
        
        # Save annotation
        ann_file = output_path / 'annotations' / f"{base_name}_annotation.json"
        with open(ann_file, 'w') as f:
            json.dump(annotation, f, indent=2)
        
        return str(ann_file)
    
    def _save_statistics(self, output_file: Path):
        """Save dataset statistics."""
        import time
        
        self.stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.stats['config'] = self.config
        
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2)


def main():
    """Command-line interface for dataset generation."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Generate 2D-3D dataset from ABC STEP files'
    )
    parser.add_argument('-i', '--input', dest='abc_path', required=True,
                       help='Path to ABC dataset STEP files')
    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                       help='Output directory for generated dataset')
    parser.add_argument('--max-files', type=int, default=150,
                       help='Maximum number of files to process')
    parser.add_argument('--num-points', type=int, default=1024,
                       help='Number of points per point cloud')
    
    args = parser.parse_args()
    
    generator = DatasetGenerator({
        'num_points': args.num_points
    })
    
    stats = generator.generate_dataset(
        args.abc_path,
        args.output_dir,
        max_files=args.max_files
    )
    
    print("\n" + "="*60)
    print("Dataset Generation Complete!")
    print("="*60)
    print(f"Total models: {stats['total_models']}")
    print(f"Successful: {stats['successful']}")
    print(f"Failed: {stats['failed']}")
    print(f"Drawings generated: {stats['drawings_generated']}")
    print(f"Point clouds generated: {stats['pointclouds_generated']}")
    print(f"Annotations generated: {stats['annotations_generated']}")


if __name__ == '__main__':
    main()
