#!/usr/bin/env python3
"""
Comprehensive Pipeline Visualization on a Single STEP File.

Визуализирует все промежуточные этапы конвейера обработки:
1. Загрузка STEP-модели.
2. Генерация 2D-чертежей (виды спереди, сверху, сбоку).
3. Генерация 3D-облаков точек (идеальное, в допуске, вне допуска).
4. Извлечение 2D-признаков с полными атрибутами Node2D.
5. Извлечение 3D-признаков с полными атрибутами Node3D.
6. Построение гибридного графа соответствий.
7. Визуализация сегментированного облака точек и сводная статистика.

Usage:
    python scripts/visualize_pipeline.py \
        -i data/abc_dataset/abc_0000_step_v00/00000000/00000000_290a9120f9f249a7a05cfe9c_step_000.step \
        -o output/visualization
"""

# ==============================================================================
# 0. GLOBAL SETUP & ENVIRONMENT CONFIGURATION
# ==============================================================================
import os
import sys
import json
import gc
import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

# Настройка LD_LIBRARY_PATH для совместимости с YOLOv7 ДО импорта остальных библиотек
conda_prefix = os.environ.get('CONDA_PREFIX')
if conda_prefix:
    ld_lib_path = os.environ.get('LD_LIBRARY_PATH', '')
    conda_lib_path = os.path.join(conda_prefix, 'lib')
    if conda_lib_path not in ld_lib_path:
        os.environ['LD_LIBRARY_PATH'] = f"{conda_lib_path}:{ld_lib_path}"

# Настройка путей проекта для корректного импорта локальных модулей
PROJECT_ROOT = Path(__file__).parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

# Импорт визуализации (до извлечения признаков, чтобы избежать конфликтов torch)
try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    print("Warning: matplotlib not available. Visualizations will be limited.")


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def _safe_get_feature_type(feature: Any) -> str:
    """
    Безопасно извлекает тип признака из словаря или строки.
    Учитывает возможные артефакты сериализации (пробелы в ключах JSON).
    """
    if isinstance(feature, dict):
        for key in ('type', 'type ', 'feature_type', 'feature_type '):
            val = feature.get(key)
            if val is not None:
                return str(val).strip()
    elif isinstance(feature, str):
        return feature.strip()
    return 'unknown'


# ==============================================================================
# MAIN CLASS
# ==============================================================================

class PipelineVisualizer:
    """
    Визуализатор полного конвейера обработки для одного STEP-файла.
    Отображает все промежуточные этапы с проверкой полноты атрибутов.
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {
            'num_points': 4096,
            'show_intermediate': True,
            'save_json': True
        }
        self.results: Dict[str, Any] = {
            'step_file': None,
            'drawings': [],
            'pointclouds': [],
            'features_2d': [],
            'features_3d': [],
            'graph': None,
            'stats': {}
        }

    def run_and_visualize(self, step_file: str, output_dir: str) -> None:
        """
        Запускает полный конвейер и визуализирует все этапы на одном холсте.
        """
        if not HAS_MATPLOTLIB:
            print("Error: matplotlib required for visualization")
            return

        step_path = Path(step_file)
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        self.results['step_file'] = str(step_file)
        
        print("=" * 70)
        print(f"Pipeline Visualization: {step_path.name}")
        print("=" * 70)

        # Создание холста: 5 строк x 3 колонки
        fig = plt.figure(figsize=(20, 25))
        fig.suptitle(f'Complete Pipeline Visualization: {step_path.name}',
                     fontsize=16, fontweight='bold')

        # ЭТАП 1: Генерация 2D-чертежей
        print("\n[Stage 1/6] Generating 2D drawings from STEP...")
        self._generate_drawings(step_path, output_path)
        self._visualize_drawings(fig, step_path, output_path)

        # ЭТАП 2: Генерация 3D-облаков точек
        print("\n[Stage 2/6] Generating 3D point clouds from STEP...")
        self._generate_pointclouds(step_path, output_path)
        self._visualize_pointclouds(fig, output_path)

        # ЭТАП 3: Извлечение 2D-признаков
        print("\n[Stage 3/6] Extracting 2D features (with full Node2D attributes)...")
        self._extract_2d_features(output_path)
        self._visualize_2d_features(fig, output_path)

        # ЭТАП 4: Извлечение 3D-признаков
        print("\n[Stage 4/6] Extracting 3D features (with full Node3D attributes)...")
        self._extract_3d_features(output_path)

        # ЭТАП 4.5: Визуализация сегментированного облака точек
        print("\n[Stage 4.5/6] Visualizing segmented point cloud...")
        self._visualize_segmented_pointcloud(fig, output_path)
        self._visualize_3d_features(fig, output_path)

        # ЭТАП 5: Построение гибридного графа
        print("\n[Stage 5/6] Building hybrid graph with correspondences...")
        self._build_graph(output_path)
        self._visualize_graph(fig, output_path)

        # ЭТАП 6: Сводка и верификация
        print("\n[Stage 6/6] Creating summary and verification...")
        self._create_summary(fig, output_path)

        # Сохранение результатов и финального изображения
        self._save_results(output_path)
        
        plt.tight_layout()
        plt.savefig(output_path / 'pipeline_visualization.png', dpi=150, bbox_inches='tight')
        plt.close()

        print("\n" + "=" * 70)
        print("  Visualization complete!")
        print(f"  Output directory: {output_path}")
        print(f"  Main visualization: {output_path / 'pipeline_visualization.png'}")
        print("=" * 70)

    # ==========================================================================
    # GENERATION STAGES
    # ==========================================================================

    def _generate_drawings(self, step_path: Path, output_path: Path) -> None:
        """ЭТАП 1: Генерация 2D-чертежей (SVG) из STEP-файла."""
        from render_with_points_utils import export_shape_to_svg_with_tolerances, read_step_file

        shape = read_step_file(str(step_path))
        views = {
            'front': {'location': (0, 0, 0), 'direction': (0, 0, 1)},
            'top': {'location': (0, 0, 0), 'direction': (0, 1, 0)},
            'side': {'location': (0, 0, 0), 'direction': (1, 0, 0)}
        }
        
        drawings_dir = output_path / 'drawings'
        drawings_dir.mkdir(parents=True, exist_ok=True)

        for view_name, config in views.items():
            svg_file = drawings_dir / f"{step_path.stem}_{view_name}.svg"
            try:
                export_shape_to_svg_with_tolerances(
                    shape,
                    filename=str(svg_file),
                    location=config['location'],
                    direction=config['direction'],
                    add_tolerances=True,
                    export_hidden_edges=False  # Отключено для экономии памяти
                )
                self.results['drawings'].append({'view': view_name, 'file': str(svg_file)})
                print(f"    Generated {view_name} view: {svg_file.name}")
            except Exception as e:
                print(f"    Failed {view_name} view: {e}")
                self.results['drawings'].append({'view': view_name, 'file': None, 'error': str(e)})

        # Конвертация SVG в PNG для совместимости с OpenCV на следующих этапах
        self._convert_svg_to_png(drawings_dir)

        # Очистка памяти после работы с тяжелой геометрикой
        del shape
        gc.collect()

    def _convert_svg_to_png(self, drawings_dir: Path) -> None:
        """ПОДЭТАП 1.1: Конвертация SVG в PNG."""
        try:
            import cairosvg
            for svg_file in drawings_dir.glob("*.svg"):
                png_file = svg_file.with_suffix('.png')
                try:
                    cairosvg.svg2png(
                        url=str(svg_file), 
                        write_to=str(png_file), 
                        scale=2,
                        background_color='white'
                    )
                    print(f"    Converted {svg_file.name} → {png_file.name}")
                except Exception as e:
                    print(f"    Failed to convert {svg_file.name}: {e}")
        except ImportError:
            print("  ⚠ cairosvg not available, SVG to PNG conversion skipped")

    def _generate_pointclouds(self, step_path: Path, output_path: Path) -> None:
        """ЭТАП 2: Генерация 3D-облаков точек с привязкой к граням (Face-aware)."""
        from features3d import load_step, generate_face_aware_pointcloud

        pc_dir = output_path / 'pointclouds'
        pc_dir.mkdir(parents=True, exist_ok=True)

        try:
            print(f"  [PC] Loading STEP for face-aware generation: {step_path}")
            shape = load_step(str(step_path))
            
            # Генерация базового облака с привязкой к граням
            points, face_ids = generate_face_aware_pointcloud(
                shape, mesh_deflection=0.1, points_per_face=500
            )
            print(f"  [PC] Generated base face-aware cloud: {len(points)} points, {len(np.unique(face_ids))} faces")

            # Создание вариантов с разным уровнем шума (отклонения)
            deviation_configs = [
                ('ideal', 0.0),    # Без шума
                ('in_tol', 0.05),  # В пределах допуска
                ('out_tol', 0.15), # Вне допуска
            ]

            for suffix, noise_level in deviation_configs:
                pc_file = pc_dir / f"{step_path.stem}_pc_{suffix}.npy"
                noisy_points = points.copy()
                
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, points.shape)
                    noisy_points += noise

                np.save(str(pc_file), noisy_points)
                self.results['pointclouds'].append({
                    'type': suffix,
                    'file': str(pc_file),
                    'num_points': len(noisy_points),
                    'bbox': {
                        'min': noisy_points.min(axis=0).tolist(),
                        'max': noisy_points.max(axis=0).tolist()
                    }
                })
                print(f"    Generated {suffix} point cloud ({len(noisy_points)} points)")
                
        except ImportError as e:
            print(f"    Failed to import features3d.py: {e}")
            print("  💡 Убедитесь, что features3d.py лежит в корне проекта")
            self.results['pointclouds'].append({'type': 'error', 'file': None, 'error': str(e)})
        except Exception as e:
            print(f"    Failed point cloud generation: {e}")
            import traceback
            traceback.print_exc()
            self.results['pointclouds'].append({'type': 'error', 'file': None, 'error': str(e)})

    def _extract_2d_features(self, output_path: Path) -> None:
        """ЭТАП 3: Извлечение 2D-признаков с полными атрибутами Node2D."""
        from feature_extraction.feature_2d import FeatureExtractor2D, save_features as save_features_2d

        extractor_config = {
            'use_bert': True,
            'use_resnet': True,
            'blur_kernel': 5,
            'morph_kernel': 3,
            'morph_iterations': 1,
            'min_area': 50,
            'confidence_threshold': 0.5,
            'model_path': '/home/spectr/itmo/recg_drawing/yolov7/weights/0324_dim_and_tol_best.pt',
            'classes': ['dimension', 'tolerance_upper', 'tolerance_lower', 'tolerance']
        }
        
        extractor = FeatureExtractor2D(extractor_config)
        features_dir = output_path / 'features_2d'
        features_dir.mkdir(parents=True, exist_ok=True)

        for drawing_info in self.results['drawings']:
            if drawing_info.get('file') is None:
                continue
                
            view = drawing_info['view']
            drawing_file = Path(drawing_info['file']).resolve()
            
            # Приоритет: PNG (для OpenCV), затем SVG
            png_file = drawing_file.with_suffix('.png')
            image_file = png_file if png_file.exists() else drawing_file
            
            if not image_file.exists():
                print(f"    No image file for {view} view")
                continue

            try:
                features = extractor.extract_features(str(image_file))
                output_file = features_dir / f"features_{view}.json"
                save_features_2d(features, str(output_file))
                
                self.results['features_2d'].append({
                    'view': view,
                    'file': str(output_file),
                    'num_features': len(features),
                    'features': [f.to_dict() if hasattr(f, 'to_dict') else f for f in features]
                })
                print(f"    Extracted {len(features)} 2D features ({view})")
            except Exception as e:
                print(f"    Failed 2D feature extraction ({view}): {e}")

    def _extract_3d_features(self, output_path: Path) -> None:
        """ЭТАП 4: Извлечение 3D-признаков на основе топологии STEP."""
        from features3d import (
            load_step, extract_faces, build_face_adjacency,
            recognize_features_topology, generate_face_aware_pointcloud,
            build_point_feature_labels, build_feature_config, save_config
        )

        features_dir = output_path / 'features_3d'
        features_dir.mkdir(parents=True, exist_ok=True)
        step_path = Path(self.results['step_file'])

        try:
            print(f"  [3D] Loading STEP: {step_path}")
            shape = load_step(str(step_path))
            
            print("  [3D] Recognizing features from B-Rep topology...")
            faces = extract_faces(shape)
            adjacency = build_face_adjacency(shape, faces)
            features_topology = recognize_features_topology(faces, adjacency)
            print(f"  [3D] Found {len(features_topology)} topological features")

            print("  [3D] Generating face-aware point cloud...")
            points, face_ids = generate_face_aware_pointcloud(shape, mesh_deflection=0.1, points_per_face=500)
            print(f"  [3D] Generated {len(points)} points with face mapping")

            # Построение маппинга: ID грани -> ID признака
            face_to_feature = {}
            for feat_idx, feat in enumerate(features_topology):
                for fid in feat["face_ids"]:
                    face_to_feature[fid] = feat_idx

            feature_labels = build_point_feature_labels(face_ids, face_to_feature)
            n_segments = len(np.unique(feature_labels[feature_labels >= 0]))
            print(f"  [3D] Mapped into {n_segments} segments")

            print("  [3D] Mapping topology to point cloud and building config...")
            config = build_feature_config(points, face_ids, features_topology, adjacency, face_to_feature, faces)

            output_file = features_dir / "features_in_tol.json"
            print(f"  Output 3d features json: {str(output_file)}")
            save_config(config, str(output_file))

            self.results['features_3d'].append({
                'type': 'in_tol',
                'file': str(output_file),
                'num_features': len(config['features']),
                'features': config['features']
            })
            print(f"    Extracted {len(config['features'])} 3D features (STEP-based)")
            
        except ImportError as e:
            print(f"    Failed to import features3d.py: {e}")
        except Exception as e:
            print(f"    Failed 3D feature extraction (STEP-based): {e}")
            import traceback
            traceback.print_exc()

    def _build_graph(self, output_path: Path) -> None:
        """ЭТАП 5: Построение гибридного графа с полными соответствиями."""
        from feature_extraction.graph_builder import construct_graph

        graphs_dir = output_path / 'graphs'
        graphs_dir.mkdir(parents=True, exist_ok=True)

        f2d_file = output_path / 'features_2d' / 'features_front.json'
        f3d_file = output_path / 'features_3d' / 'features_in_tol.json'
        output_file = graphs_dir / 'hybrid_graph.json'

        if f2d_file.exists() and f3d_file.exists():
            try:
                print("  [Graph] Starting hybrid graph construction...")
                graph = construct_graph(str(f2d_file), str(f3d_file), str(output_file))
                
                self.results['graph'] = {
                    'file': str(output_file),
                    'num_2d_nodes': len(graph.nodes_2d),
                    'num_3d_nodes': len(graph.nodes_3d),
                    'num_edges': len(graph.edges),
                    'edges': graph.edges
                }
                print(f"    Built hybrid graph ({len(graph.edges)} edges)")
            except Exception as e:
                print(f"    Failed graph construction: {e}")
                import traceback
                traceback.print_exc()
        else:
            print("    Missing feature files for graph construction")
            if not f2d_file.exists():
                print(f"     Missing: {f2d_file}")
            if not f3d_file.exists():
                print(f"     Missing: {f3d_file}")

    # ==========================================================================
    # VISUALIZATION STAGES
    # ==========================================================================

    def _visualize_drawings(self, fig: plt.Figure, step_path: Path, output_path: Path) -> None:
        """Визуализация 2D-чертежей (Подграфик 1-3)."""
        drawings_dir = output_path / 'drawings'
        for idx, view_name in enumerate(['front', 'top', 'side']):
            ax = fig.add_subplot(5, 3, idx + 1)
            svg_file = drawings_dir / f"{step_path.stem}_{view_name}.svg"
            
            if svg_file.exists():
                try:
                    import cairosvg
                    from io import BytesIO
                    from PIL import Image
                    
                    png_data = cairosvg.svg2png(url=str(svg_file))
                    img = Image.open(BytesIO(png_data))
                    ax.imshow(img)
                    ax.set_title(f'{view_name.capitalize()} View (from STEP)', fontsize=12, fontweight='bold')
                except ImportError:
                    ax.text(0.5, 0.5, f'{view_name}\n(SVG: {svg_file.name})', ha='center', va='center', fontsize=12)
                    ax.set_title(f'{view_name.capitalize()} View', fontsize=12)
            else:
                ax.text(0.5, 0.5, 'Failed', ha='center', va='center')
                ax.set_title(f'{view_name.capitalize()} View (Failed)', fontsize=12, color='red')
            ax.axis('off')

    def _visualize_pointclouds(self, fig: plt.Figure, output_path: Path) -> None:
        """Визуализация 3D-облаков точек с разным уровнем шума (Подграфик 4-6)."""
        pc_dir = output_path / 'pointclouds'
        for idx, pc_info in enumerate(self.results['pointclouds']):
            ax = fig.add_subplot(5, 3, idx + 4, projection='3d')
            pc_file = Path(pc_info['file']) if pc_info.get('file') else None
            
            if pc_file and pc_file.exists():
                points = np.load(pc_file)
                colors = {'ideal': 'blue', 'in_tol': 'green', 'out_tol': 'red'}
                color = colors.get(pc_info['type'], 'gray')
                
                # Субдискретизация для ускорения рендеринга
                if len(points) > 500:
                    indices = np.random.choice(len(points), 500, replace=False)
                    points = points[indices]
                    
                ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=color, s=3, alpha=0.6)
                ax.set_title(f'{pc_info["type"]} ({pc_info["num_points"]} pts)\nfrom STEP', fontsize=10, fontweight='bold')
                ax.set_xlabel('X')
                ax.set_ylabel('Y')
                ax.set_zlabel('Z')
            else:
                ax.text(0, 0, 0, 'Failed', ha='center', va='center')
                ax.set_title(f'{pc_info["type"]} (Failed)', fontsize=10, color='red')

    def _visualize_segmented_pointcloud(self, fig: plt.Figure, output_path: Path) -> None:
        """Визуализация сегментированного 3D-облака точек по типам признаков (Подграфик 7)."""
        features_dir = output_path / 'features_3d'
        feat_file = features_dir / 'features_in_tol.json'
        ax = fig.add_subplot(5, 3, 7, projection='3d')

        if not feat_file.exists():
            ax.text(0, 0, 0, 'Not extracted', ha='center', va='center')
            ax.set_title('Segmented Point Cloud (Failed)', fontsize=10, color='red')
            return

        try:
            with open(feat_file, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError:
            ax.text(0, 0, 0, 'JSON Error\n(See Stage 4 Log)', ha='center', va='center', color='red')
            return

        features = data.get('features', [])
        if not features:
            ax.text(0, 0, 0, 'No features', ha='center', va='center')
            return

        # Загрузка исходного облака точек
        pc_dir = output_path / 'pointclouds'
        step_name = Path(self.results['step_file']).stem
        pc_files = list(pc_dir.glob(f"{step_name}*_in_tol.npy")) or list(pc_dir.glob("*_in_tol.npy"))
        
        if not pc_files:
            ax.text(0, 0, 0, 'Point cloud not found', ha='center', va='center')
            return

        all_points = np.load(str(pc_files[0].resolve()))
        
        # Словарь цветов для типов признаков
        type_colors = {
            'hole': 'red', 'slot': 'orange', 'pocket': 'purple', 'chamfer': 'yellow',
            'fillet': 'pink', 'step': 'brown', 'island': 'cyan', 'counterbore': 'magenta',
            'countersink': 'lime', 'taper_hole': 'olive', 'plane': 'blue', 'cylinder': 'green',
            'cone': 'darkorange', 'sphere': 'gold', 'torus': 'violet', 'unknown': 'gray'
        }
        segment_colors = ['red', 'green', 'blue', 'orange', 'purple', 'cyan', 'magenta', 
                          'yellow', 'brown', 'pink', 'lime', 'olive', 'teal', 'coral', 'navy']

        plotted_mask = np.zeros(len(all_points), dtype=bool)
        
        # Сортировка по количеству точек (сначала мелкие, чтобы они не перекрывались крупными)
        sorted_features = sorted(features, key=lambda f: len(f.get('point_indices', [])))
        
        # Извлечение типов для проверки однородности
        feature_types = [_safe_get_feature_type(f) for f in sorted_features]
        all_same_type = len(set(feature_types)) == 1

        type_counts: Dict[str, int] = {}
        segment_labels: List[str] = []

        for seg_idx, feature in enumerate(sorted_features):
            point_indices = feature.get('point_indices', feature.get('nodes', [])) if isinstance(feature, dict) else []
            feature_type = _safe_get_feature_type(feature)

            if not point_indices:
                continue

            point_indices = np.array(point_indices, dtype=int)
            
            # Исключаем уже отрисованные точки для избежания наложения
            try:
                unique_indices = point_indices[~plotted_mask[point_indices]]
            except IndexError:
                unique_indices = point_indices

            if len(unique_indices) > 0:
                try:
                    seg_points = all_points[unique_indices]
                except IndexError:
                    seg_points = all_points

                # Выбор цвета: по сегменту (если все типы одинаковы) или по типу признака
                if all_same_type:
                    color = segment_colors[seg_idx % len(segment_colors)]
                    label = f'Segment {seg_idx+1} ({feature_type}, {len(unique_indices)}pts)'
                else:
                    color = type_colors.get(feature_type, 'gray')
                    label = f'{feature_type} ({len(unique_indices)}pts)'

                ax.scatter(seg_points[:, 0], seg_points[:, 1], seg_points[:, 2],
                           c=color, s=8, alpha=0.8, label=label, edgecolors='none')
                segment_labels.append(label)

                try:
                    plotted_mask[unique_indices] = True
                except IndexError:
                    plotted_mask = np.ones(len(all_points), dtype=bool)

                type_counts[feature_type] = type_counts.get(feature_type, 0) + len(unique_indices)

        # Настройка заголовка и легенды
        title_stats = ', '.join(f'{k}:{v}pts' for k, v in type_counts.items())
        ax.set_title(f'Segmented Point Cloud ({len(features)} segments)\n{title_stats}', fontsize=10, fontweight='bold')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        if type_counts:
            if all_same_type:
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=segment_colors[i % len(segment_colors)],
                                      markersize=8, label=segment_labels[i])
                           for i in range(len(segment_labels))]
            else:
                non_zero_types = {k: v for k, v in type_counts.items() if v > 0}
                handles = [plt.Line2D([0], [0], marker='o', color='w',
                                      markerfacecolor=type_colors.get(t, 'gray'),
                                      markersize=8, label=f'{t} ({c}pts)')
                           for t, c in non_zero_types.items()]
            ax.legend(handles=handles, loc='upper right', fontsize=7)

    def _visualize_2d_features(self, fig: plt.Figure, output_path: Path) -> None:
        """Визуализация 2D-признаков: гистограмма и детекции YOLOv7 (Подграфики 8 и 15)."""
        features_dir = output_path / 'features_2d'
        feat_file = features_dir / 'features_front.json'

        # Подграфик 8: Распределение типов 2D-признаков
        ax = fig.add_subplot(5, 3, 8)
        if feat_file.exists():
            with open(feat_file, 'r') as f:
                data = json.load(f)
            features = data.get('features', [])
            
            type_counts: Dict[str, int] = {}
            for feat in features:
                ft = feat.get('semantic_info', {}).get('feature_category', feat.get('type', 'unknown'))
                type_counts[ft] = type_counts.get(ft, 0) + 1

            if type_counts:
                types = list(type_counts.keys())
                counts = list(type_counts.values())
                bars = ax.bar(types, counts, color=['blue', 'red', 'green', 'orange'][:len(types)])
                ax.set_title(f'2D Features (Front): {len(features)} total', fontsize=10, fontweight='bold')
                ax.set_xlabel('Feature Category')
                ax.set_ylabel('Count')
                plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
                for bar, val in zip(bars, counts):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.05,
                            str(val), ha='center', va='bottom', fontsize=9)
            else:
                ax.text(0.5, 0.5, 'No features', ha='center', va='center')
        else:
            ax.text(0.5, 0.5, 'Not extracted', ha='center', va='center')
            ax.set_title('2D Features (Failed)', fontsize=10, color='red')

        # Подграфик 15: Визуализация детекций YOLOv7 с OCR на исходном изображении
        ax2 = fig.add_subplot(5, 3, 15)
        drawings_dir = output_path / 'drawings'
        
        # Динамическое определение имени файла вместо жесткого хардкода
        step_stem = Path(self.results['step_file']).stem
        front_png = drawings_dir / f"{step_stem}_front.png"

        if front_png.exists() and feat_file.exists():
            import cv2
            img = cv2.imread(str(front_png))
            if img is not None:
                with open(feat_file, 'r') as f:
                    data = json.load(f)
                features = data.get('features', [])
                
                img_with_boxes = img.copy()
                for i, feat in enumerate(features[:6]):  # Ограничение до 6 для читаемости
                    bbox = feat.get('bbox', [0, 0, 50, 50])
                    x1, y1, x2, y2 = map(int, bbox)
                    cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    text = feat.get('text_content', '')[:15]
                    label = f"#{i+1}: {text}"
                    cv2.putText(img_with_boxes, label, (x1, y1-5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

                img_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
                ax2.imshow(img_rgb)
                ax2.set_title(f'YOLOv7 Detections + OCR ({len(features)} features)', fontsize=10, fontweight='bold')
                ax2.axis('off')
            else:
                ax2.text(0.5, 0.5, 'Image not loaded', ha='center', va='center')
                ax2.set_title('OCR Preprocessing (Failed)', fontsize=10, color='red')
        else:
            ax2.text(0.5, 0.5, 'Not available', ha='center', va='center')
            ax2.set_title('OCR Preprocessing (Failed)', fontsize=10, color='red')

    def _visualize_3d_features(self, fig: plt.Figure, output_path: Path) -> None:
        """Визуализация распределения типов 3D-признаков (Подграфик 9)."""
        features_dir = output_path / 'features_3d'
        ax = fig.add_subplot(5, 3, 9)
        feat_file = features_dir / 'features_in_tol.json'

        if feat_file.exists():
            try:
                with open(feat_file, 'r') as f:
                    data = json.load(f)
                
                features = data.get('features', [])
                if isinstance(features, dict):
                    features = list(features.values())
                elif not isinstance(features, list):
                    features = []

                type_counts: Dict[str, int] = {}
                for feat in features:
                    if not isinstance(feat, dict):
                        continue
                    ft = _safe_get_feature_type(feat)
                    type_counts[ft] = type_counts.get(ft, 0) + 1

                if type_counts:
                    types = list(type_counts.keys())
                    counts = list(type_counts.values())
                    ax.pie(counts, labels=types, autopct='%1.1f%%', startangle=90)
                    ax.set_title(f'3D Features (in_tol): {len(features)} total\nfrom STEP point cloud',
                                 fontsize=10, fontweight='bold')
                else:
                    ax.text(0.5, 0.5, 'No valid features found', ha='center', va='center')
            except Exception as e:
                ax.text(0.5, 0.5, f'Error: {str(e)}', ha='center', va='center', color='red')
        else:
            ax.text(0.5, 0.5, 'Not extracted', ha='center', va='center')
            ax.set_title('3D Features (Failed)', fontsize=10, color='red')

    def _visualize_graph(self, fig: plt.Figure, output_path: Path) -> None:
        """Визуализация статистики гибридного графа и статусов допусков (Подграфики 10 и 11)."""
        # Подграфик 10: Общая статистика графа
        ax = fig.add_subplot(5, 3, 10)
        if self.results['graph']:
            graph = self.results['graph']
            stats = [graph['num_2d_nodes'], graph['num_3d_nodes'], graph['num_edges']]
            labels = ['2D Nodes', '3D Nodes', 'Edges']
            colors = ['blue', 'green', 'orange']
            
            bars = ax.bar(labels, stats, color=colors)
            ax.set_title('Hybrid Graph Statistics', fontsize=12, fontweight='bold')
            ax.set_ylabel('Count')
            for bar, val in zip(bars, stats):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        str(val), ha='center', va='bottom', fontsize=10)
        else:
            ax.text(0.5, 0.5, 'Graph not built', ha='center', va='center')
            ax.set_title('Hybrid Graph (Failed)', fontsize=12, color='red')

        # Подграфик 11: Статус допусков ребер
        ax2 = fig.add_subplot(5, 3, 11)
        if self.results['graph']:
            edges = self.results['graph'].get('edges', [])
            if edges:
                status_counts: Dict[str, int] = {}
                for edge in edges:
                    status = edge.get('tolerance_status', 'UNKNOWN')
                    status_counts[status] = status_counts.get(status, 0) + 1
                
                labels = list(status_counts.keys())
                counts = list(status_counts.values())
                colors = ['green', 'orange', 'red', 'gray'][:len(labels)]
                
                ax2.pie(counts, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
                ax2.set_title('Correspondence Tolerance Status', fontsize=12, fontweight='bold')
            else:
                ax2.text(0.5, 0.5, 'No edges', ha='center', va='center')
        else:
            ax2.text(0.5, 0.5, 'No data', ha='center', va='center')

    def _create_summary(self, fig: plt.Figure, output_path: Path) -> None:
        """Создание комплексной сводной панели (Подграфики 12, 13, 14)."""
        # Подграфик 12: Сводка конвейера
        ax = fig.add_subplot(5, 3, 12)
        ax.axis('off')
        summary_text = "Pipeline Summary\n" + "=" * 55 + "\n"
        summary_text += f"STEP File:\n{Path(self.results['step_file']).name}\n"
        
        summary_text += "\n2D Drawings (from STEP):\n"
        for drawing in self.results['drawings']:
            status = " " if drawing.get('file') else " "
            summary_text += f"  {status} {drawing['view']}\n"
            
        summary_text += "\n3D Point Clouds (from STEP):\n"
        for pc in self.results['pointclouds']:
            status = " " if pc.get('file') else " "
            pts = pc.get('num_points', 'N/A')
            summary_text += f"  {status} {pc['type']}: {pts} points\n"
            
        summary_text += "\n2D Features:\n"
        for feat in self.results['features_2d']:
            summary_text += f"    {feat['view']}: {feat['num_features']} features\n"
            
        summary_text += "\n3D Features:\n"
        for feat in self.results['features_3d']:
            summary_text += f"    {feat['type']}: {feat['num_features']} features\n"
            
        if self.results['graph']:
            g = self.results['graph']
            summary_text += f"\nHybrid Graph:\n"
            summary_text += f"  2D Nodes: {g['num_2d_nodes']}\n"
            summary_text += f"  3D Nodes: {g['num_3d_nodes']}\n"
            summary_text += f"  Edges: {g['num_edges']}\n"

        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=9,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        # Подграфик 13: Чек-лист верификации атрибутов
        ax2 = fig.add_subplot(5, 3, 13)
        ax2.axis('off')
        checklist_text = "Attribute Verification (Full Structures)\n" + "=" * 45 + "\n"
        checklist_text += "Node2D (from 2D drawings):\n"
        checklist_text += "    id, type, value, tolerance\n"
        checklist_text += "    position_2d, semantic_info\n"
        checklist_text += "    text_content, confidence, bbox\n"
        checklist_text += "Node3D (from STEP point clouds):\n"
        checklist_text += "    id, feature_type, measured_value\n"
        checklist_text += "    position_3d, normals, curvature\n"
        checklist_text += "    topology, geometry_features\n"
        checklist_text += "    spatial_encoding, point_indices\n"
        checklist_text += "EdgeCorrespondence:\n"
        checklist_text += "    source_id, target_id, weight\n"
        checklist_text += "    deviation, tolerance_status\n"
        checklist_text += "    spatial/semantic/geometric similarity\n"

        ax2.text(0.05, 0.95, checklist_text, transform=ax2.transAxes,
                 fontsize=8, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))

        # Подграфик 14: Список выходных файлов
        ax3 = fig.add_subplot(5, 3, 14)
        ax3.axis('off')
        output_text = "Output Files (from STEP)\n" + "=" * 45 + "\n"
        output_text += "  drawings/           - 2D SVG (from STEP)\n"
        output_text += "  pointclouds/        - 3D NPY (from STEP)\n"
        output_text += "  features_2d/        - Node2D JSON\n"
        output_text += "  features_3d/        - Node3D JSON\n"
        output_text += "  graphs/             - Hybrid graph JSON\n"
        output_text += "  pipeline_results.json\n"

        ax3.text(0.05, 0.95, output_text, transform=ax3.transAxes,
                 fontsize=9, verticalalignment='top', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

    def _save_results(self, output_path: Path) -> None:
        """Сохранение всех собранных метаданных конвейера в JSON."""
        results_file = output_path / 'pipeline_results.json'
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"    Saved results to {results_file.name}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main() -> None:
    """Точка входа в скрипт с обработкой аргументов командной строки."""
    parser = argparse.ArgumentParser(
        description='Visualize complete pipeline on a single STEP file'
    )
    parser.add_argument('-i', '--input', dest='step_file', required=True,
                        help='Input STEP file')
    parser.add_argument('-o', '--output', dest='output_dir', required=True,
                        help='Output directory for visualizations')
    parser.add_argument('--num-points', type=int, default=4096,
                        help='Number of points per point cloud')
    
    args = parser.parse_args()

    visualizer = PipelineVisualizer({
        'num_points': args.num_points
    })
    visualizer.run_and_visualize(args.step_file, args.output_dir)


if __name__ == '__main__':
    main()