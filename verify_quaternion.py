import os  
import glob  
import json  
import numpy as np  
import cv2  # Используем OpenCV для рисования и работы с изображениями  
from scipy.spatial.transform import Rotation as R  
from tqdm import tqdm  

# --- НАСТРОЙКИ ---  

# Путь к вашей папке с сгенерированным датасетом  
# Он должен содержать папки 'annotations' и 'views'  
DATASET_DIR = "c:/Users/walle/Downloads/archive-2025-12-19_21-03-13/archive"     

# Папка для сохранения результатов проверки  
OUTPUT_DIR = os.path.join(DATASET_DIR, "verification_output")  
os.makedirs(OUTPUT_DIR, exist_ok=True)  

# Параметры камеры (должны СОВПАДАТЬ с параметрами из abc_pipline.py)  
IMG_SIZE = (512, 512)  
FOV_DEG = 60.0  

# Длина осей для визуализации (можете сделать их больше/меньше)  
# Поскольку модели нормализованы в куб [-0.5, 0.5], длина 0.5 будет хорошо видна  
AXIS_LENGTH = 0.5  

# --- ОСНОВНЫЕ ФУНКЦИИ ---  

def quaternion_to_rotation_matrix(quat_wxyz):  
    """  
    Конвертирует кватернион формата [w, x, y, z] в матрицу вращения 3x3.  
    """  
    # SciPy ожидает формат [x, y, z, w]  
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]  
    return R.from_quat(quat_xyzw).as_matrix()  

def project_points(points_3d, view_matrix, projection_matrix, img_width, img_height):  
    """  
    Проецирует 3D точки на 2D изображение.  
    Это упрощенная версия функции из вашего основного скрипта.  
    """  
    # Объединяем матрицы  
    mvp_matrix = projection_matrix @ view_matrix  

    # Добавляем гомогенную координату  
    points_3d_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))  

    # Проецируем  
    points_clip = (mvp_matrix @ points_3d_h.T).T  

    # Перспективное деление  
    w = points_clip[:, 3]  
    valid_indices = w > 1e-5  
    if not np.any(valid_indices):  
        return None  
        
    points_ndc = points_clip[valid_indices, :3] / w[valid_indices, np.newaxis]  

    # Преобразование в экранные координаты  
    points_2d = np.zeros((len(points_ndc), 2), dtype=int)  
    points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * img_width  
    points_2d[:, 1] = (1.0 - (points_ndc[:, 1] + 1.0) * 0.5) * img_height  

    return points_2d  

def build_matrices_from_view(view_info, fov_deg, width, height):  
    """  
    Воссоздает матрицы вида и проекции на основе информации из аннотации.  
    """  
    # 1. Восстанавливаем матрицу вида из кватерниона и позиции камеры  
    cam_pos = np.array(view_info['camera_pos'])  
    orientation_quat = view_info['orientation_quaternion']  
    
    # Кватернион описывает вращение из мировых координат в координаты камеры.  
    # Поэтому мы можем напрямую построить матрицу вращения.  
    rotation_mat = quaternion_to_rotation_matrix(orientation_quat)  
    
    view_matrix = np.eye(4)  
    view_matrix[:3, :3] = rotation_mat  
    # Часть с трансляцией: -R * t  
    view_matrix[:3, 3] = -rotation_mat @ cam_pos  

    # 2. Строим матрицу проекции (должна быть идентична той, что в генераторе)  
    fov_y_rad = np.deg2rad(fov_deg)  
    aspect_ratio = width / height  
    near_clip, far_clip = 0.01, 100.0  
    f = 1.0 / np.tan(fov_y_rad / 2.0)  
    
    projection_matrix = np.array([  
        [f / aspect_ratio, 0, 0, 0],  
        [0, f, 0, 0],  
        [0, 0, (far_clip + near_clip) / (near_clip - far_clip), (2 * far_clip * near_clip) / (near_clip - far_clip)],  
        [0, 0, -1, 0]  
    ])  

    return view_matrix, projection_matrix  

def draw_axes(image, origin_2d, axes_2d):  
    """Рисует оси на изображении."""  
    # Цвета: BGR для OpenCV  
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)] # X=красный, Y=зеленый, Z=синий  
    for point, color in zip(axes_2d, colors):  
        cv2.line(image, tuple(origin_2d), tuple(point), color, 2)  
    return image  

# --- ГЛАВНЫЙ ЦИКЛ ---  

if __name__ == '__main__':  
    print(f"--- Starting Quaternion Verification ---")  
    print(f"Reading data from: {DATASET_DIR}")  
    print(f"Saving output to: {OUTPUT_DIR}")  

    annotation_files = glob.glob(os.path.join(DATASET_DIR, 'annotations', '*.json'))  

    if not annotation_files:  
        print("ERROR: No annotation files found. Did you run the generation script first?")  
        exit()  

    # Создаем 3D оси координат  
    # Начало в (0,0,0), и три точки на концах осей X, Y, Z  
    origin_3d = np.array([[0, 0, 0]])  
    axes_3d = np.array([  
        [AXIS_LENGTH, 0, 0],  # Конец оси X  
        [0, AXIS_LENGTH, 0],  # Конец оси Y  
        [0, 0, AXIS_LENGTH]   # Конец оси Z  
    ])  

    for ann_path in tqdm(annotation_files, desc="Processing models"):  
        with open(ann_path, 'r') as f:  
            annotations = json.load(f)  

        for view in annotations['views']:  
            # Загружаем оригинальное изображение  
            img_path = os.path.join(DATASET_DIR, view['img_path'])  
            if not os.path.exists(img_path):  
                continue  
            
            image = cv2.imread(img_path)  
            
            # Воссоздаем матрицы вида и проекции  
            view_matrix, proj_matrix = build_matrices_from_view(  
                view, FOV_DEG, IMG_SIZE[0], IMG_SIZE[1]  
            )  

            # Проецируем начало координат и концы осей  
            origin_2d = project_points(origin_3d, view_matrix, proj_matrix, IMG_SIZE[0], IMG_SIZE[1])  
            axes_2d = project_points(axes_3d, view_matrix, proj_matrix, IMG_SIZE[0], IMG_SIZE[1])  

            if origin_2d is None or axes_2d is None:  
                continue  

            # Рисуем оси на изображении  
            image_with_axes = draw_axes(image, origin_2d[0], axes_2d)  

            # Сохраняем результат  
            model_name = annotations['model_name']  
            view_name = view['view_name']  
            output_filename = f"{model_name}_{view_name}_verification.png"  
            output_path = os.path.join(OUTPUT_DIR, output_filename)  
            cv2.imwrite(output_path, image_with_axes)  

    print("\n--- Verification complete! Check the 'verification_output' folder. ---")