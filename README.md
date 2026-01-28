# YOLOv8_3DFusion: Method for Enriching 2D Representations with 3D Point Cloud Data

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.13](https://img.shields.io/badge/Python-3.13-green.svg)](https://www.python.org/downloads/release/python-3130/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-red.svg)](https://pytorch.org/)

**This repository contains the implementation of a novel method for enriching 2D representations with geometric data from 3D point clouds to improve scene understanding in industrial applications.**

![Prediction with axes](docs/images/Prediction_with_axes.png)

## Overview

The project addresses a critical gap in industrial computer vision systems: the lack of methods that effectively integrate visual semantics from 2D images with geometric precision from 3D data. Our approach, **YOLOv8_3DFusion**, enables simultaneous interpretation of technical documentation (2D) and physical object geometry (3D), significantly improving orientation estimation accuracy.

## Key Features

- **Two-level fusion architecture**: Combines global context and regional features for accurate orientation estimation
- **SOTA results**: Reduces angular error from 57.9° to 17.5° while maintaining high detection accuracy (mAP@0.5 = 0.89)
- **Industrial applicability**: Designed specifically for manufacturing and CAD analysis scenarios
- **Extensible**: Framework supports integration of GNN-based feature detectors

## Project Structure
```
YOLOv8_3DFusion/
├── docs/ # Documentation and presentation
├── src/ # Source code
│ ├── models/ # Model implementations
│ ├── utils/ # Utility functions
│ └── train_fusion.py # Training script
├── scripts/ # Inference and visualization scripts
├── data/ # Synchronized dataset
├── results/ # Results and visualizations
├── configs/ # Configuration files
└── ... # Other supporting files
```

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/your-username/YOLOv8_3DFusion.git
cd YOLOv8_3DFusion
```
2. Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # Linux/MacOS
venv\Scripts\activate    # Windows
pip install -r requirements.txt
```

### Training
```bash
python src/train_fusion.py
```

### Dataset Verification
```bash
python scripts/verify_quaternion.py
```
## Results

Method | Data Type |mAP@0.5 | Angular Error | 
--- | --- | --- | --- |
YOLOv8 (baseline) | RGB | 0.92 | ~57.9° | 
3D-only (ICP + GT bbox) | Dense point cloud | - | ~22-25° | 
YOLOv8_3DFusion | RGB + sparse point cloud | 0.89 | 17.5° 
