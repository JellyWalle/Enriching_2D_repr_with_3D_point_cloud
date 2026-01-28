import os  


os.environ["PYGLET_HEADLESS"] = "1"
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"

os.environ["PYGLET_HEADLESS"] = "1"

import pyglet
print("Pyglet headless:", pyglet.options['headless'])
print("Display:", pyglet.canvas.get_display())

import glob  
import json  
import numpy as np  
import trimesh  
from PIL import Image  
from tqdm import tqdm  
import io  
import traceback   
from PIL import Image, ImageDraw  
from scipy.spatial.transform import Rotation as R  
import time

# --- Константы ---  
IMG_SIZE = (512, 512)  
FOV_DEG = 60.0  
N_VIEWS = 2 # Количество ракурсов для каждой модели  

# --- Пути ---  
ROOT_DIR = "/mnt/hdd_works/ml_640/yolov11ir/abc_full_10k_v00/10k"  
INPUT_DATA_DIR = os.path.join(ROOT_DIR, "train", "512")  
OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset_generated_final") # Новая папка для "чистого" результата  

# --- Количество моделей для обработки ---  
NUM_MODELS_TO_PROCESS = -1

# --- Создание директорий ---  
os.makedirs(OUTPUT_DIR, exist_ok=True)  
for subdir in ['views', 'points', 'annotations', 'views_debug']:  
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)

def load_and_normalize_mesh(obj_path):  
    """  
    Загружает, объединяет, центрирует, исправляет и нормализует 3D-модель.  
    Возвращает None в случае ошибки.  
    """  
    try:  
        # Загружаем, отключаем авто-обработку  
        mesh = trimesh.load(obj_path, force='mesh', process=False) 
        if isinstance(mesh, trimesh.Scene):  
            if not mesh.geometry: return None  
            # Объединяем все геометрии в одну, если это сцена  
            mesh = mesh.dump(concatenate=True)  
        if not hasattr(mesh, 'vertices') or not mesh.vertices.size: return None  

        # Принудительно исправляем нормали, чтобы избежать черных рендеров  
        mesh.fix_normals()  

        # Центрируем модель в начале координат  
        mesh.vertices -= mesh.bounds.mean(axis=0)  

        # Нормализуем размер модели, чтобы она вписывалась в единичный куб  
        max_extent = np.max(mesh.extents)  
        if max_extent < 1e-6: return None # Пропускаем вырожденные меши  
        mesh.vertices /= max_extent  

        return mesh  
    except Exception:  
        return None  

def get_camera_positions(n_views, radius, y_pos_ratio=0.3):  
    """Создает равномерно распределенные позиции камер на окружности."""  
    positions = []   
    y_pos = radius * y_pos_ratio  
    for i in range(n_views):  
        theta = (i / n_views) * 2 * np.pi  
        x = radius * np.cos(theta)  
        z = radius * np.sin(theta)  
        positions.append(np.array([x, y_pos, z]))  
    return positions

def get_random_camera_positions(n_views, radius, min_height_ratio=0.1, max_height_ratio=0.5):  
    """  
    Создает случайные позиции камер на поверхности сферы.  
    """  
    positions = []  
    for _ in range(n_views):  
        # Случайный азимут (угол в плоскости XY) от 0 до 2*PI  
        theta = np.random.uniform(0, 2 * np.pi)  
        
        # Случайный угол возвышения (от полюса).  
        # Чтобы точки были распределены равномерно по площади сферы,  
        # нужно использовать arccos от равномерного распределения.  
        # Ограничим phi, чтобы камера не смотрела строго снизу или сверху.  
        # Например, от 30 до 80 градусов от вертикали.  
        phi_min_rad = np.arccos(max_height_ratio) # Угол от вертикали  
        phi_max_rad = np.arccos(min_height_ratio)  
        phi = np.random.uniform(phi_min_rad, phi_max_rad)  

        # Конвертируем сферические координаты в декартовы  
        x = radius * np.sin(phi) * np.cos(theta)  
        y = radius * np.cos(phi) # Высота  
        z = radius * np.sin(phi) * np.sin(theta)  
        
        # Можно добавить небольшое случайное смещение к радиусу, чтобы имитировать приближение/отдаление  
        random_radius = radius * np.random.uniform(0.9, 1.1)  
        
        positions.append(np.array([x, y, z]) * (random_radius / radius))  
    return positions 

def get_random_up_vector(view_direction):  
    """  
    Генерирует случайный "up" вектор, который перпендикулярен направлению взгляда.  
    Это позволяет камере вращаться вокруг своей оси взгляда (создавать "крен").  
    """  
    # 1. Находим любой вектор, не коллинеарный направлению взгляда.  
    #    Если взгляд почти вертикален, берем ось X. Иначе берем ось Y.  
    if np.abs(np.dot(view_direction, [0, 1, 0])) > 0.99:  
        v_aux = np.array([1, 0, 0]) # Взгляд почти вертикальный, используем ось X  
    else:  
        v_aux = np.array([0, 1, 0]) # В остальных случаях используем ось Y  

    # 2. Создаем первую ось перпендикулярную взгляду (векторное произведение)  
    right_vector = np.cross(view_direction, v_aux)  
    right_vector /= np.linalg.norm(right_vector)  

    # 3. Теперь у нас есть направление взгляда и вектор "вправо".  
    #    Их векторное произведение даст нам вектор "вверх", который 100% перпендикулярен взгляду.  
    #    Это наш "стандартный" up-вектор.  
    standard_up = np.cross(right_vector, view_direction)  
    standard_up /= np.linalg.norm(standard_up)  
    
    # 4. Теперь вращаем этот стандартный "up" вектор на случайный угол вокруг оси взгляда.  
    random_roll_angle = np.random.uniform(0, 2 * np.pi)  
    
    # Матрица вращения вокруг оси (формула Родрига)  
    K = np.array([  
        [0, -view_direction[2], view_direction[1]],  
        [view_direction[2], 0, -view_direction[0]],  
        [-view_direction[1], view_direction[0], 0]  
    ])  
    R = np.eye(3) + np.sin(random_roll_angle) * K + (1 - np.cos(random_roll_angle)) * (K @ K)  
    
    # Применяем вращение к стандартному up-вектору  
    random_up = R @ standard_up  
    
    return random_up  

def look_at_matrix(eye, target, up):  
    """Создает матрицу вида (View Matrix) world-to-camera."""  
    eye, target, up = np.asarray(eye), np.asarray(target), np.asarray(up)  
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

def matrix_to_quaternion(view_matrix):  
    """  
    Конвертирует матрицу вида (world-to-camera) в кватернион ориентации.  
    Кватернион описывает вращение, которое нужно применить к объекту в мировой  
    системе координат, чтобы он "смотрел" в камеру.  
    Возвращает кватернион в формате [w, x, y, z].  
    """  
    # Матрица вида (view matrix) показывает, как преобразовать мир к камере.  
    # Матрица вращения объекта относительно камеры является ее инверсией.  
    # Но для простоты мы можем работать напрямую с матрицей вращения из view_matrix.  
    # Верхний левый 3x3 блок view_matrix - это и есть матрица вращения.  
    rotation_matrix = view_matrix[:3, :3]  
    
    # Создаем объект Rotation из матрицы  
    r = R.from_matrix(rotation_matrix)  
    
    # Получаем кватернион в формате [x, y, z, w]  
    quat_xyzw = r.as_quat()  
    
    # Конвертируем в более стандартный формат [w, x, y, z] и возвращаем как список  
    quat_wxyz = [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]  
    
    return quat_wxyz  

def process_one_model(obj_path, n_views):  
    """  
    Полный цикл обработки одной 3D-модели с генерацией bbox и корректной отладочной визуализацией.  
    """  
    model_name = os.path.splitext(os.path.basename(obj_path))[0]  

    mesh = load_and_normalize_mesh(obj_path)  
    if mesh is None:  
        print("Mesh is none")
        return

    bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1))  
    optimal_radius = bounding_radius / np.tan(np.deg2rad(FOV_DEG / 2.0)) * 1.8  

    pc_path = os.path.join(OUTPUT_DIR, 'points', f'{model_name}.npy')  
    np.save(pc_path, mesh.vertices)  

    #cam_positions = get_camera_positions(n_views, radius=optimal_radius)  
    cam_positions = get_random_camera_positions(n_views, radius=optimal_radius)  

    annotations = {  
        'model_name': model_name,  
        'point_cloud_path': os.path.relpath(pc_path, OUTPUT_DIR),  
        'views': []  
    }  

    for i, cam_pos in enumerate(cam_positions):  
        try:
            scene = trimesh.Scene(mesh)
            view_name = f'view_{i:03d}'  

            # 0. Генерируем случайный UP вектор для этого направления  
            # Вычисляем направление взгляда  
            target_pos = np.array([0, 0, 0])  
            view_direction = target_pos - cam_pos  
            view_direction /= np.linalg.norm(view_direction)  
            
            up_vector = get_random_up_vector(view_direction)  
            #up_vector = [0, 1, 0]

            view_matrix_wc = look_at_matrix(eye=cam_pos, target=[0, 0, 0], up=up_vector)  
            scene.camera_transform = np.linalg.inv(view_matrix_wc)  
            scene.camera.resolution = IMG_SIZE  
            scene.camera.fov = (FOV_DEG, FOV_DEG)  

            # 1. Рендерим чистое изображение и получаем данные  
            #try:  

            png_data = scene.save_image()  
            if not png_data or len(png_data) == 0:
                print(f"Empty render for {model_name}, view {i}")
                continue

            # Конвертируем в объект PIL Image  
            img_pil = Image.open(io.BytesIO(png_data)).convert("RGB")  
            
            # Проверяем, что изображение не пустое/черное  
            if np.max(np.array(img_pil)) < 10:  
                print(f"{view_name}: empty frame")
                continue  

            # 2. Проецируем точки  
            points_2d, bbox_2d = project_points_to_image(scene, mesh.vertices)  
            if bbox_2d is None:  
                print(f"{view_name} :bbox is none")
                continue # Пропускаем, если рамка невалидна  

            # 3. Сохраняем чистое изображение  
            img_path = os.path.join(OUTPUT_DIR, 'views', f'{model_name}_{view_name}.png')  
            img_pil.save(img_path)

            # 4. Создаем и сохраняем отладочное изображение  
            # Создаем копию для рисования  
            img_debug_pil = img_pil.copy()  
            draw = ImageDraw.Draw(img_debug_pil)  

            # Рисуем bounding box (зеленым цветом, толщиной в 1 пиксель)  
            draw.rectangle(bbox_2d, outline="green", width=1)  

            # Рисуем спроецированные точки (красным цветом)  
            if points_2d is not None and len(points_2d) > 0:  
                # Преобразуем координаты в кортежи для PIL  
                drawable_points = [tuple(p) for p in points_2d]  
                draw.point(drawable_points, fill="red")  

            # Сохраняем отладочное изображение  
            img_debug_path = os.path.join(OUTPUT_DIR, 'views_debug', f'{model_name}_{view_name}_debug.png')  
            img_debug_pil.save(img_debug_path)  
            orientation_quaternion = matrix_to_quaternion(view_matrix_wc)  

            time.sleep(0.1)

            # 5. Сохраняем аннотацию  
            view_info = {  
                'view_name': view_name,  
                'img_path': os.path.relpath(img_path, OUTPUT_DIR),  
                'camera_pos': cam_pos.tolist(),  
                'bbox_2d': bbox_2d,  
                'orientation_quaternion': orientation_quaternion,  
                'class_id': 0,  
                'class_name': 'part'  
            }  
            annotations['views'].append(view_info)  
            del scene
            import gc
            gc.collect()
        except Exception as e:  
            # Ловим редкие ошибки на уровне ракурса и пропускаем его, а не всю модель  
            print(f"\n  - Warning: Skipping view {i} for model {model_name} due to error: {e}")  
            continue  

    if annotations['views']:  
        ann_path = os.path.join(OUTPUT_DIR, 'annotations', f'{model_name}.json')  
        with open(ann_path, 'w') as f:  
            json.dump(annotations, f, indent=2)  


# --- НОВАЯ ФУНКЦИЯ ---  
def project_points_to_image(scene, points_3d):  
    """  
    Проецирует 3D точки на 2D изображение с использованием параметров камеры из сцены.  
    Использует корректную матрицу проекции.  
    Возвращает 2D точки и bounding box.  
    """  
    # 1. Получаем матрицу вида (World-to-Camera)  
    # Инвертируем camera-to-world матрицу из trimesh  
    view_matrix = np.linalg.inv(scene.camera_transform)  

    # 2. Строим матрицу проекции (Camera-to-Clip Space)  
    # Используем fov и разрешение, как в OpenGL  
    width, height = scene.camera.resolution  
    fov_y = np.deg2rad(scene.camera.fov[1]) # fov по вертикали  
    aspect_ratio = width / height  
    near_clip = 0.01  
    far_clip = 100.0 # Эти значения можно настроить, но для нормализованных моделей они подходят  

    f = 1.0 / np.tan(fov_y / 2.0)  
    
    projection_matrix = np.array([  
        [f / aspect_ratio, 0, 0, 0],  
        [0, f, 0, 0],  
        [0, 0, (far_clip + near_clip) / (near_clip - far_clip), (2 * far_clip * near_clip) / (near_clip - far_clip)],  
        [0, 0, -1, 0]  
    ])  

    # 3. Объединяем матрицы в одну: Model-View-Projection (MVP)  
    mvp_matrix = projection_matrix @ view_matrix  

    # 4. Применяем MVP матрицу к точкам  
    # Добавляем гомогенную координату к 3D точкам  
    points_3d_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))  
    # Проецируем  
    points_clip_space = (mvp_matrix @ points_3d_h.T).T  

    # 5. Деление на W (перспективная коррекция) для перехода в NDC (-1 to 1)  
    # Избегаем деления на ноль или очень малые числа  
    w = points_clip_space[:, 3]  
    # Отбираем только те точки, что находятся перед камерой и имеют корректный W  
    valid_indices = w > 1e-4  
    if not np.any(valid_indices):  
        return None, None  
        
    points_ndc = points_clip_space[valid_indices, :3] / w[valid_indices, np.newaxis]


    # 6. Преобразование из NDC в экранные координаты (Screen Space)  
    # Координаты X и Y в NDC находятся в диапазоне [-1, 1]  
    # Координата Z в NDC (глубина) также в [-1, 1]  
    # Отсекаем точки вне экрана (с небольшим запасом)  
    in_screen = (np.abs(points_ndc[:, 0]) < 1.0) & (np.abs(points_ndc[:, 1]) < 1.0)  
    if not np.any(in_screen):  
        return None, None  
        
    points_ndc_screen = points_ndc[in_screen]  

    points_2d = np.zeros((len(points_ndc_screen), 2))  
    points_2d[:, 0] = (points_ndc_screen[:, 0] + 1.0) * 0.5 * width  
    points_2d[:, 1] = (1.0 - (points_ndc_screen[:, 1] + 1.0) * 0.5) * height # Y инвертирован в экранных координатах  

    # 7. Вычисляем bounding box из видимых точек  
    x_min, y_min = points_2d.min(axis=0)  
    x_max, y_max = points_2d.max(axis=0)  

    # Ограничиваем рамку границами изображения  
    x_min = max(0, x_min)  
    y_min = max(0, y_min)  
    x_max = min(width - 1, x_max)  
    y_max = min(height - 1, y_max)  
    
    if x_max <= x_min or y_max <= y_min:  
        return points_2d, None # Возвращаем точки, но невалидный bbox  

    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]  

    return points_2d, bbox  

if __name__ == '__main__':  
    print("--- Starting Robust Dataset Generation ---")  
    obj_files = glob.glob(os.path.join(INPUT_DATA_DIR, '**', '*.obj'), recursive=True)  

    if NUM_MODELS_TO_PROCESS > 0:  
        obj_files_subset = obj_files[:NUM_MODELS_TO_PROCESS]  
    else:  
        obj_files_subset = obj_files # Обрабатываем все найденные файлы  

    print(f"Found {len(obj_files)} models. Processing {len(obj_files_subset)}.")  
    print(f"Output will be saved to: {OUTPUT_DIR}")  

    # Основной цикл с прогресс-баром  
    for obj_path in tqdm(obj_files_subset, desc="Processing models"):  
        try:  
            process_one_model(obj_path, n_views=N_VIEWS)  
        except Exception as e:  
            # Логгируем фатальные, непредвиденные ошибки для конкретной модели  
            print(f"\n---!!! UNEXPECTED FATAL Error on {os.path.basename(obj_path)}: {e} !!!---")  
            traceback.print_exc()  

    print("\n--- Dataset generation complete. ---")
