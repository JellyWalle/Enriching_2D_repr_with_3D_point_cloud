#!/usr/bin/env python3
"""
Complete Pipeline for Hybrid GNN Dataset Generation.

This script runs the full pipeline:
1. Generate 2D drawings and 3D point clouds from STEP files
2. Extract 2D features (dimensions, tolerances)
3. Extract 3D features (manufacturing features)
4. Build hybrid graphs with correspondences
5. Split into train/val/test sets

Usage:
    python scripts/run_pipeline.py \
        -i /path/to/abc_dataset \
        -o /path/to/output \
        --max-files 150
"""

import os
import sys
import json
import shutil
from pathlib import Path
from typing import Dict, List
import argparse

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from data_generation.generate_data import DatasetGenerator
from feature_extraction.feature_2d import FeatureExtractor2D, save_features as save_features_2d
from feature_extraction.feature_3d import FeatureExtractor3D, save_features as save_features_3d
from feature_extraction.graph_builder import GraphConstructor, save_graph


class HybridGNNDatasetPipeline:
    """
    Complete pipeline for hybrid GNN dataset generation.
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or {
            'max_files': 150,
            'num_points': 1024,
            'train_ratio': 0.70,
            'val_ratio': 0.20,
            'test_ratio': 0.10
        }
        
        self.stats = {
            'models_processed': 0,
            'features_2d_extracted': 0,
            'features_3d_extracted': 0,
            'graphs_constructed': 0,
            'train_samples': 0,
            'val_samples': 0,
            'test_samples': 0
        }
    
    def run_pipeline(self, abc_dataset_path: str,
                    output_dir: str) -> Dict:
        """
        Run complete pipeline.
        
        Args:
            abc_dataset_path: Path to ABC dataset STEP files
            output_dir: Output directory for all generated data
            
        Returns:
            Pipeline statistics
        """
        abc_path = Path(abc_dataset_path)
        output_path = Path(output_dir)
        
        print("="*60)
        print("Hybrid GNN Dataset Generation Pipeline")
        print("="*60)
        
        # Step 1: Generate base dataset
        print("\n[Step 1/5] Generating base dataset (drawings + point clouds)...")
        self._generate_base_dataset(abc_path, output_path)
        
        # Step 2: Extract 2D features
        print("\n[Step 2/5] Extracting 2D features from drawings...")
        self._extract_2d_features(output_path)
        
        # Step 3: Extract 3D features
        print("\n[Step 3/5] Extracting 3D features from point clouds...")
        self._extract_3d_features(output_path)
        
        # Step 4: Build hybrid graphs
        print("\n[Step 4/5] Building hybrid graphs...")
        self._build_graphs(output_path)
        
        # Step 5: Split into train/val/test
        print("\n[Step 5/5] Splitting into train/val/test sets...")
        self._create_splits(output_path)
        
        # Save final statistics
        self._save_statistics(output_path / 'pipeline_stats.json')
        
        print("\n" + "="*60)
        print("Pipeline Complete!")
        print("="*60)
        print(f"Models processed: {self.stats['models_processed']}")
        print(f"2D features extracted: {self.stats['features_2d_extracted']}")
        print(f"3D features extracted: {self.stats['features_3d_extracted']}")
        print(f"Graphs constructed: {self.stats['graphs_constructed']}")
        print(f"Train samples: {self.stats['train_samples']}")
        print(f"Val samples: {self.stats['val_samples']}")
        print(f"Test samples: {self.stats['test_samples']}")
        
        return self.stats
    
    def _generate_base_dataset(self, abc_path: Path, output_path: Path):
        """Generate 2D drawings and 3D point clouds."""
        generator = DatasetGenerator({
            'num_points': self.config['num_points']
        })
        
        stats = generator.generate_dataset(
            str(abc_path),
            str(output_path),
            max_files=self.config['max_files']
        )
        
        self.stats['models_processed'] = stats['successful']
        print(f"  Generated {stats['drawings_generated']} drawings")
        print(f"  Generated {stats['pointclouds_generated']} point clouds")
    
    def _extract_2d_features(self, output_path: Path):
        """Extract 2D features from all drawings."""
        extractor = FeatureExtractor2D()
        annotations_dir = output_path / 'annotations'
        features_dir = output_path / 'processed' / 'features_2d'
        features_dir.mkdir(parents=True, exist_ok=True)
        
        total_features = 0
        
        for ann_file in annotations_dir.glob('*.json'):
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            # Extract features from each drawing
            for drawing_file in annotation.get('drawings', []):
                if os.path.exists(drawing_file):
                    features = extractor.extract_features(drawing_file)
                    
                    # Save features
                    model_id = ann_file.stem.replace('_annotation', '')
                    view = Path(drawing_file).stem.split('_')[-1]
                    output_file = features_dir / f"{model_id}_{view}_features.json"
                    save_features_2d(features, str(output_file))
                    
                    total_features += len(features)
        
        self.stats['features_2d_extracted'] = total_features
        print(f"  Extracted {total_features} 2D features")
    
    def _extract_3d_features(self, output_path: Path):
        """Extract 3D features from all point clouds."""
        extractor = FeatureExtractor3D()
        annotations_dir = output_path / 'annotations'
        features_dir = output_path / 'processed' / 'features_3d'
        features_dir.mkdir(parents=True, exist_ok=True)
        
        total_features = 0
        
        for ann_file in annotations_dir.glob('*.json'):
            with open(ann_file, 'r') as f:
                annotation = json.load(f)
            
            # Extract features from each point cloud
            for pc_file in annotation.get('pointclouds', []):
                if os.path.exists(pc_file):
                    features = extractor.extract_features(pc_file)
                    
                    # Save features
                    model_id = ann_file.stem.replace('_annotation', '')
                    pc_type = Path(pc_file).stem.split('_pc_')[-1]
                    output_file = features_dir / f"{model_id}_{pc_type}_features.json"
                    save_features_3d(features, str(output_file))
                    
                    total_features += len(features)
        
        self.stats['features_3d_extracted'] = total_features
        print(f"  Extracted {total_features} 3D features")
    
    def _build_graphs(self, output_path: Path):
        """Build hybrid graphs from 2D and 3D features."""
        constructor = GraphConstructor()
        features_2d_dir = output_path / 'processed' / 'features_2d'
        features_3d_dir = output_path / 'processed' / 'features_3d'
        graphs_dir = output_path / 'graphs'
        graphs_dir.mkdir(parents=True, exist_ok=True)
        
        graphs_built = 0
        
        # Match 2D and 3D features by model ID
        model_ids = set()
        for f in features_2d_dir.glob('*_features.json'):
            model_ids.add(f.stem.rsplit('_', 1)[0])
        
        for model_id in model_ids:
            # Find matching 2D and 3D feature files
            f2d_files = list(features_2d_dir.glob(f"{model_id}_*_features.json"))
            f3d_files = list(features_3d_dir.glob(f"{model_id}_*_features.json"))
            
            if not f2d_files or not f3d_files:
                continue
            
            # Build graph for each view
            for f2d_file in f2d_files:
                view = f2d_file.stem.split('_')[-2]  # front/top/side
                
                # Find matching 3D file (use in_tolerance)
                f3d_file = features_3d_dir / f"{model_id}_in_tol_features.json"
                
                if not f3d_file.exists():
                    continue
                
                # Construct graph
                try:
                    graph = constructor.construct_graph(
                        str(f2d_file),
                        str(f3d_file)
                    )
                    
                    # Save graph
                    output_file = graphs_dir / f"{model_id}_{view}_graph.json"
                    save_graph(graph, str(output_file))
                    graphs_built += 1
                except Exception as e:
                    print(f"  Warning: Could not build graph for {model_id}_{view}: {e}")
        
        self.stats['graphs_constructed'] = graphs_built
        print(f"  Built {graphs_built} hybrid graphs")
    
    def _create_splits(self, output_path: Path):
        """Split graphs into train/val/test sets."""
        graphs_dir = output_path / 'graphs'
        processed_dir = output_path / 'processed'
        
        # Get all graph files
        graph_files = list(graphs_dir.glob('*_graph.json'))
        
        # Shuffle and split
        import random
        random.shuffle(graph_files)
        
        n_total = len(graph_files)
        n_train = int(n_total * self.config['train_ratio'])
        n_val = int(n_total * self.config['val_ratio'])
        n_test = n_total - n_train - n_val
        
        # Create splits
        splits = {
            'train': graph_files[:n_train],
            'val': graph_files[n_train:n_train + n_val],
            'test': graph_files[n_train + n_val:]
        }
        
        # Copy files to split directories
        for split_name, files in splits.items():
            split_dir = processed_dir / split_name
            split_dir.mkdir(parents=True, exist_ok=True)
            
            for graph_file in files:
                shutil.copy(str(graph_file), str(split_dir / graph_file.name))
            
            # Save split manifest
            manifest = {
                'num_files': len(files),
                'files': [f.name for f in files]
            }
            with open(split_dir / 'manifest.json', 'w') as f:
                json.dump(manifest, f, indent=2)
        
        self.stats['train_samples'] = n_train
        self.stats['val_samples'] = n_val
        self.stats['test_samples'] = n_test
        
        print(f"  Train: {n_train} samples")
        print(f"  Val: {n_val} samples")
        print(f"  Test: {n_test} samples")
    
    def _save_statistics(self, output_file: Path):
        """Save pipeline statistics."""
        import time
        
        self.stats['timestamp'] = time.strftime('%Y-%m-%d %H:%M:%S')
        self.stats['config'] = self.config
        
        with open(output_file, 'w') as f:
            json.dump(self.stats, f, indent=2)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Complete pipeline for hybrid GNN dataset generation'
    )
    parser.add_argument('-i', '--input', dest='abc_path', required=True,
                       help='Path to ABC dataset STEP files')
    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                       help='Output directory for all generated data')
    parser.add_argument('--max-files', type=int, default=150,
                       help='Maximum number of files to process')
    parser.add_argument('--num-points', type=int, default=1024,
                       help='Number of points per point cloud')
    
    args = parser.parse_args()
    
    pipeline = HybridGNNDatasetPipeline({
        'max_files': args.max_files,
        'num_points': args.num_points
    })
    
    stats = pipeline.run_pipeline(args.abc_path, args.output_dir)
    
    print(f"\nOutput saved to: {args.output_dir}")


if __name__ == '__main__':
    main()
