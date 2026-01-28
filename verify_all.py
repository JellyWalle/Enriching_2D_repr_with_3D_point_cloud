import os
import glob
import json
import numpy as np
import trimesh
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm

# --- НАСТРОЙКИ ---
DATASET_DIR = "c:/Users/walle/Downloads/archive-2025-12-19_21-03-13/archive"
OUTPUT_DIR = os.path.join(DATASET_DIR, "verification_rotated_pointclouds")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Длина осей для визуализации
AXIS_LENGTH = 0.5

def quaternion_to_rotation_matrix(quat_wxyz):
    """Конвертирует [w, x, y, z] → матрица 3x3."""
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return R.from_quat(quat_xyzw).as_matrix()

def visualize_rotated_pointcloud(pc_path, quat_wxyz, output_path):
    """
    Поворачивает облако точек по кватерниону и визуализирует с осями.
    """
    try:
        # 1. Загрузка облака
        points = np.load(pc_path)
        if points.size == 0:
            print(f"Warning: Empty point cloud at {pc_path}")
            return

        # 2. Применение поворота
        R_mat = quaternion_to_rotation_matrix(quat_wxyz)
        points_rotated = (R_mat @ points.T).T  # [N, 3]

        # 3. Создание облака и осей
        point_cloud = trimesh.points.PointCloud(points_rotated)
        
        # Оси координат (в повёрнутой СК)
        axes = trimesh.creation.axis(axis_length=AXIS_LENGTH)

        # 4. Сцена
        scene = trimesh.Scene([point_cloud, axes])
        
        # Фиксированный ракурс (вид спереди)
        camera_pos = np.array([2.0, 0.0, 1.5])
        target_pos = np.zeros(3)
        up_vector = np.array([0.0, 0.0, 1.0])
        
        view_matrix = look_at_matrix(eye=camera_pos, target=target_pos, up=up_vector)
        scene.camera.transform = np.linalg.inv(view_matrix)

        # 5. Сохранение
        png_data = scene.save_image(resolution=(800, 600))
        with open(output_path, 'wb') as f:
            f.write(png_data)

    except Exception as e:
        print(f"Error processing {pc_path}: {e}")

def look_at_matrix(eye, target, up):
    """Создаёт view matrix (world-to-camera)."""
    eye, target, up = map(np.asarray, (eye, target, up))
    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-6: return np.eye(4)
    f /= f_norm

    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6: return np.eye(4)
    s /= s_norm

    u = np.cross(s, f)

    view_matrix = np.eye(4)
    view_matrix[0, :3] = s
    view_matrix[1, :3] = u
    view_matrix[2, :3] = -f
    view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], eye)
    return view_matrix

# --- ОСНОВНОЙ ЦИКЛ ---
if __name__ == '__main__':
    print(f"--- Visualizing Rotated Point Clouds ---")
    print(f"Dataset: {DATASET_DIR}")
    print(f"Output: {OUTPUT_DIR}")

    annotation_files = glob.glob(os.path.join(DATASET_DIR, 'annotations', '*.json'))
    if not annotation_files:
        raise FileNotFoundError("No annotation files found!")

    # Ограничим для быстрой проверки
    N_DEBUG = 100
    annotation_files = annotation_files[:N_DEBUG]

    for ann_path in tqdm(annotation_files, desc="Rotating Point Clouds"):
        with open(ann_path, 'r') as f:
            ann = json.load(f)

        # Берём ПЕРВУЮ view (или можно все)
        view = ann['views'][0]
        quat = view['orientation_quaternion']
        pc_path = os.path.join(DATASET_DIR, ann['point_cloud_path'])
        if not os.path.exists(pc_path):
            continue

        model_name = ann['model_name']
        output_path = os.path.join(OUTPUT_DIR, f"{model_name}_rotated.png")
        
        if not os.path.exists(output_path):
            visualize_rotated_pointcloud(pc_path, quat, output_path)

    print("\n--- Done! Check 'verification_rotated_pointclouds' folder. ---")