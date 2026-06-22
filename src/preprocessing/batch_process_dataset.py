#!/usr/bin/env python3
"""
Полный пайплайн пакетной обработки датасета ABC для обучения HybridGNN.
Генерирует для каждого STEP-файла:
1. 2D-чертежи (Front, Top, Side) в SVG и PNG.
2. 2D-признаки (features_front.json, features_top.json, features_side.json).
3. 3D-облака точек с шумом (pointcloud_in_tol.ply, pointcloud_out_tol.ply).
4. 3D-признаки с топологией и геометрией (features_in_tol.json, features_out_tol.json).
5. HTML-визуализацию сегментации (features_3d_in_tol.html, features_3d_out_tol.html).
6. Гибридные графы соответствий (graph_in_tol.json, graph_out_tol.json).
"""
import os
import sys
import json
import argparse
import numpy as np
import open3d as o3d
from pathlib import Path
from typing import Dict, List
import gc

# ==============================================================================
# 1. НАСТРОЙКА ПУТЕЙ (точно как в visualize_pipeline.py)
# ==============================================================================
# Если скрипт лежит в src/preprocessing/, то parent.parent.parent = корень проекта
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / 'src'))
sys.path.insert(0, str(PROJECT_ROOT / 'utils'))

print(f" Project Root set to: {PROJECT_ROOT}")

# ==============================================================================
# 2. ИМПОРТЫ
# ==============================================================================
# 3D модуль (лежит в корне проекта)
try:
    from features3d import (
        load_step, extract_faces, build_face_adjacency,
        recognize_features_topology, generate_face_aware_pointcloud,
        build_feature_config, visualize_features, save_config
    )
    print("  features3d.py успешно импортирован")
except ImportError as e:
    print(f"   Ошибка импорта features3d.py: {e}")
    sys.exit(1)

# 2D модуль
try:
    from feature_extraction.feature_2d import FeatureExtractor2D, save_features as save_features_2d
    HAS_2D_EXTRACTOR = True
    print("  2D экстрактор успешно импортирован")
except ImportError:
    HAS_2D_EXTRACTOR = False
    print("    2D экстрактор не найден. Убедитесь, что src/feature_extraction/feature_2d.py существует.")

# Графостроитель
try:
    from feature_extraction.graph_builder import construct_graph
    HAS_GRAPH_BUILDER = True
    print("  Graph builder успешно импортирован")
except ImportError:
    HAS_GRAPH_BUILDER = False
    print("    Graph builder не найден. Убедитесь, что src/feature_extraction/graph_builder.py существует.")

# Конфигурация шума
NOISE_CONFIGS = {
    'in_tol': 0.05,   # В пределах допуска (позитивные пары)
    'out_tol': 0.15   # Вне допуска (негативные пары)
}

# Генератор чертежей
try:
    from render_with_points_utils import export_shape_to_svg_with_tolerances, ToleranceManager
    HAS_DRAWING_GENERATOR = True
    print("  Drawing generator успешно импортирован")
except ImportError:
    HAS_DRAWING_GENERATOR = False
    print("    Drawing generator не найден. Убедитесь, что render_with_points_utils.py существует.")


import signal

class TimeoutException(Exception):
    pass

def _timeout_handler(signum, frame):
    raise TimeoutException(" Operation timed out!")

def run_with_timeout(func, timeout_sec, *args, **kwargs):
    """
    Выполняет функцию с жестким таймаутом.
    Если функция не успела выполниться за timeout_sec секунд, прерывает её.
    Работает только в главном потоке на Unix-системах.
    """
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(timeout_sec)
    try:
        result = func(*args, **kwargs)
        signal.alarm(0)  # Отменяем таймер при успехе
        return result
    except TimeoutException:
        print(f"          TIMEOUT: Функция {func.__name__} прервана по таймауту ({timeout_sec}с)")
        return None
    except Exception as e:
        signal.alarm(0)
        raise e
    finally:
        signal.signal(signal.SIGALRM, old_handler)

# ==============================================================================
# 3. ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ
# ==============================================================================
def export_svg_to_png_safe(svg_path: Path, png_path: Path, timeout_sec=30):
    """Конвертирует SVG в PNG с таймаутом (cairosvg может виснуть на битых SVG)."""
    import cairosvg
    def _convert():
        cairosvg.svg2png(url=str(svg_path), write_to=str(png_path), scale=2, background_color='white')
    
    return run_with_timeout(_convert, timeout_sec)

def limit_image_size(png_path: Path, max_dim=1024):
    """
    Уменьшает изображение, если оно слишком большое.
    На CPU YOLO будет обрабатывать 4K-картинку вечность.
    """
    try:
        from PIL import Image
        img = Image.open(png_path)
        if max(img.size) > max_dim:
            img.thumbnail((max_dim, max_dim), Image.Resampling.LANCZOS)
            img.save(png_path)
            print(f"      Image resized to {img.size} for faster CPU inference")
    except ImportError:
        pass  # Если PIL нет, просто пропускаем
    except Exception as e:
        print(f"          Failed to resize image: {e}")

# ==============================================================================
# 4. ОБРАБОТКА ОДНОГО ФАЙЛА
# ============================================================================== 
def process_single_step(step_path: Path, output_dir: Path, extractor_2d: FeatureExtractor2D):
    """Обрабатывает один STEP-файл и сохраняет все артефакты."""
    print(f"\n{'='*70}")
    print(f"Processing: {step_path.name}")
    print(f"{'='*70}")
    
    part_name = step_path.stem
    part_output_dir = output_dir / part_name
    part_output_dir.mkdir(parents=True, exist_ok=True)
    shape = None  # Для корректного удаления в finally
    
    try:
        shape = load_step(str(step_path))
        
        # =========================================================================
        # ШАГ 1: Генерация 2D-чертежей (Front, Top, Side) с КОНСИСТЕНТНЫМИ допусками
        # =========================================================================
        print("  [1/6] Generating 2D drawings with CONSISTENT tolerances...")
        drawings_dir = part_output_dir / 'drawings'
        drawings_dir.mkdir(exist_ok=True)
        
        views = {
            'front': (0, 0, 1),  # Front view: смотрим вдоль оси Z
            'top': (0, 1, 0),    # Top view: смотрим вдоль оси Y
            'side': (1, 0, 0)    # Side view: смотрим вдоль оси X
        }
        
        if HAS_DRAWING_GENERATOR:
            # Создаем ОДИН ToleranceManager для всех видов
            tolerance_manager = ToleranceManager(seed=42)
            
            for view_name, view_direction in views.items():
                svg_file = drawings_dir / f"{part_name}_{view_name}.svg"
                png_file = drawings_dir / f"{part_name}_{view_name}.png"
                
                # НОВЫЙ ВЫЗОВ: используем view_direction вместо location/direction
                def _gen_svg():
                    export_shape_to_svg_with_tolerances(
                        shape, filename=str(svg_file), direction=view_direction,
                        width=800, height=600, tolerance_manager=tolerance_manager
                    )
                #result = run_with_timeout(_gen_svg, timeout_sec=600)
                export_shape_to_svg_with_tolerances(
                        shape, filename=str(svg_file), direction=view_direction,
                        width=800, height=600, tolerance_manager=tolerance_manager
                    )
                #if result is None:
                #    print(f"        Skipped {view_name} view due to HLR timeout")
                #    continue
                import cairosvg
                cairosvg.svg2png(url=str(svg_file), write_to=str(png_file), scale=2, background_color='white')
                #if not export_svg_to_png_safe(svg_file, png_file, timeout_sec=600):
                #    print(f"        Skipped PNG conversion for {view_name} due to timeout")
                #    continue
                
                # Ограничиваем размер PNG для быстрого инференса на CPU
                limit_image_size(png_file, max_dim=1024)
                print(f"      Saved {view_name} view")
            
            # Сохраняем кэш допусков для верификации
            tolerance_cache_file = part_output_dir / 'tolerance_cache.json'
            
            # Конвертируем numpy-типы перед сохранением
            def convert_numpy(obj):
                if isinstance(obj, dict): return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
                elif isinstance(obj, list): return [convert_numpy(i) for i in obj]
                elif isinstance(obj, np.integer): return int(obj)
                elif isinstance(obj, np.floating): return float(obj)
                elif isinstance(obj, np.ndarray): return obj.tolist()
                return obj
            
            cache_to_save = convert_numpy(tolerance_manager.tolerance_cache)
            with open(tolerance_cache_file, 'w') as f:
                json.dump(cache_to_save, f, indent=2)
            print(f"      Saved tolerance cache for verification")
        else:
            print("        Drawing generator not available. Skipping 2D drawings.")

        # =========================================================================
        # ШАГ 2: Извлечение 2D-признаков (для всех 3-х видов)
        # =========================================================================
        if HAS_2D_EXTRACTOR:
            print("  [2/6] Extracting 2D features (Front, Top, Side)...")
            features_2d_dir = part_output_dir / 'features_2d'
            features_2d_dir.mkdir(exist_ok=True)
            
            #extractor = FeatureExtractor2D({
            #    'use_bert': True, 'use_resnet': True,
            #    'model_path': '/home/spectr/itmo/recg_drawing/yolov7/weights/0324_dim_and_tol_best.pt',
            #    'classes': ['dimension', 'tolerance_upper', 'tolerance_lower', 'tolerance']
            #})
            
            for view_name in views.keys():
                png_file = drawings_dir / f"{part_name}_{view_name}.png"
                if png_file.exists():
                    try:
                        def _extract():
                            return extractor_2d.extract_features(str(png_file))
                    
                        features = run_with_timeout(_extract, timeout_sec=300)
                        if features is not None:
                            out_file = features_2d_dir / f"features_{view_name}.json"
                            save_features_2d(features, str(out_file))
                            print(f"     Extracted {len(features)} 2D features ({view_name})")
                        else:
                            print(f"        Skipped {view_name} 2D extraction due to timeout")
                    except Exception as e:
                        print(f"    Failed 2D extraction ({view_name}): {e}")

        # =========================================================================
        # ШАГ 3: Распознавание 3D-топологии (один раз для всех уровней шума)
        # =========================================================================
        print("  [3/6] Recognizing 3D topology from B-Rep...")
        faces = extract_faces(shape)
        adjacency = build_face_adjacency(shape, faces)
        features_topology = recognize_features_topology(faces, adjacency)
        print(f"      Found {len(features_topology)} topological features")
        
        face_to_feature = {}
        for feat_idx, feat in enumerate(features_topology):
            for fid in feat["face_ids"]:
                face_to_feature[fid] = feat_idx

        # =========================================================================
        # ШАГ 4: Генерация облаков точек, признаков и визуализации (in_tol / out_tol)
        # =========================================================================
        print("  [4/6] Generating noisy point clouds, features, and HTML visualizations...")
        
        base_points, base_face_ids = generate_face_aware_pointcloud(
            shape, mesh_deflection=0.1, points_per_face=500
        )
        
        for suffix, noise_level in NOISE_CONFIGS.items():
            print(f"    -> Processing {suffix} (noise_sigma={noise_level})...")
            
            noisy_points = base_points.copy()
            if noise_level > 0:
                noisy_points += np.random.normal(0, noise_level, base_points.shape)
                
            ply_file = part_output_dir / f"pointcloud_{suffix}.ply"
            npy_file = part_output_dir / f"pointcloud_{suffix}.npy"
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(noisy_points)
            o3d.io.write_point_cloud(str(ply_file), pcd)
            np.save(str(npy_file), noisy_points)
            print(f"        Saved pointcloud_{suffix}.ply & .npy ({len(noisy_points)} pts)")
            
            config = build_feature_config(
                points=noisy_points,
                face_ids=base_face_ids,
                features=features_topology,
                adjacency=adjacency,
                face_to_feature=face_to_feature,
                faces=faces
            )
            
            features_file = part_output_dir / f"features_{suffix}.json"
            save_config(config, str(features_file))
            print(f"        Saved features_{suffix}.json ({len(config['features'])} features)")
            
            html_file = part_output_dir / f"features_3d_{suffix}.html"
            labels = np.full(len(noisy_points), -1, dtype=int)
            for idx, feat in enumerate(config["features"]):
                labels[feat["point_indices"]] = idx
                
            visualize_features(noisy_points, labels, config["features"], filename=str(html_file))
            print(f"        Saved features_3d_{suffix}.html")

        # =========================================================================
        # ШАГ 5: Построение гибридных графов
        # =========================================================================
        if HAS_GRAPH_BUILDER:
            print("  [5/6] Building hybrid graphs...")
            graphs_dir = part_output_dir / 'graphs'
            graphs_dir.mkdir(exist_ok=True)
            
            f2d_file = part_output_dir / 'features_2d' / 'features_front.json'
            
            for suffix in NOISE_CONFIGS.keys():
                f3d_file = part_output_dir / f"features_{suffix}.json"
                graph_file = graphs_dir / f"graph_{suffix}.json"
                
                if f2d_file.exists() and f3d_file.exists():
                    try:
                        graph = construct_graph(str(f2d_file), str(f3d_file), str(graph_file))
                        print(f"      Saved graph_{suffix}.json ({len(graph.edges)} edges)")
                    except Exception as e:
                        print(f"      Failed to build graph_{suffix}: {e}")

        print(f"  Successfully processed {part_name}")
        return True

    except Exception as e:
        print(f"   CRITICAL ERROR processing {step_path.name}: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:

        del shape
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description='Full batch processing of ABC dataset for HybridGNN')
    parser.add_argument('--input_dir', type=str, required=True, help='Root directory with STEP files')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed data')
    parser.add_argument('--limit', type=int, default=None, help='Limit processing to N files (for testing)')
    
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)


    extractor_2d = FeatureExtractor2D({
        'use_bert': True, 'use_resnet': True,
        'model_path': '/home/spectr/itmo/recg_drawing/yolov7/weights/0324_dim_and_tol_best.pt',
        'classes': ['dimension', 'tolerance_upper', 'tolerance_lower', 'tolerance']
    })
    
    step_files = list(input_path.rglob('*.step')) + list(input_path.rglob('*.stp'))
    print(f"Found {len(step_files)} STEP files in {input_path}")
    
    if args.limit:
        step_files = step_files[:args.limit]
        print(f" LIMITED TO FIRST {args.limit} FILES FOR TESTING")
        
    success_count, fail_count, skip_count = 0, 0, 0
    
    for step_file in step_files:
        # ПРОВЕРКА: существует ли уже папка с именем этого файла (без расширения) в output_path
        target_dir = output_path / step_file.stem
        
        if target_dir.exists() and target_dir.is_dir():
            print(f" Пропуск: {step_file.name} (уже обработан, папка '{target_dir.name}' существует)")
            skip_count += 1
            continue  # Переходим к следующему файлу
            
        # Если папки нет, обрабатываем файл
        if process_single_step(step_file, output_path, extractor_2d=extractor_2d):
            success_count += 1
        else:
            fail_count += 1
            
    print("\n" + "="*70)
    print("BATCH PROCESSING COMPLETE")
    print(f"  Successful: {success_count}")
    print(f"   Failed: {fail_count}")
    print(f"     Skipped (already processed): {skip_count}")
    print(f" Output saved to: {output_path}")
    print("="*70)

if __name__ == '__main__':
    main()