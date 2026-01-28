import os  
import glob  
import json  
import numpy as np  
import trimesh  
from tqdm import tqdm  


# Путь к папке с сгенерированным датасетом  
DATASET_DIR = "c:/Users/walle/Downloads/archive-2025-12-19_21-03-13/archive"  

# Папка для сохранения результатов проверки облаков точек  
OUTPUT_DIR = os.path.join(DATASET_DIR, "verification_pointclouds")  
os.makedirs(OUTPUT_DIR, exist_ok=True)  


def visualize_point_cloud(pc_path, output_path):  
    """  
    Загружает облако точек, создает 3D-визуализацию и сохраняет ее как изображение.  
    """  
    try:  
        # 1. Загружаем облако точек  
        points = np.load(pc_path)  
        if points.size == 0:  
            print(f"Warning: Point cloud at {pc_path} is empty.")  
            return  

        # 2. Создаем объект Trimesh PointCloud  
        point_cloud = trimesh.points.PointCloud(points)  

        # 3. Добавляем оси координат  
        axes = trimesh.creation.axis(axis_length=0.7)  

        # 4. Создаем сцену и добавляем в нее облако и оси  
        scene = trimesh.Scene([point_cloud, axes])  
        
        # 5. Устанавливаем ракурс камеры
        camera_pos = np.array([2.0, 2.0, 2.0])  
        target_pos = np.array([0.0, 0.0, 0.0])  
        up_vector  = np.array([0.0, 1.0, 0.0])  

        # Создаем матрицу вида (из мировых координат в координаты камеры)  
        view_matrix = look_at_matrix(eye=camera_pos, target=target_pos, up=up_vector)  
        
        # Trimesh ожидает обратную матрицу (из координат камеры в мировые)  
        camera_transform = np.linalg.inv(view_matrix)  
        
        # Применяем трансформацию к камере сцены  
        scene.camera.transform = camera_transform  

        # 6. Сохраняем сцену как изображение  
        png_data = scene.save_image(resolution=(800, 600))  
        
        # 7. Записываем данные в файл  
        with open(output_path, 'wb') as f:  
            f.write(png_data)  

    except Exception as e:  
        print(f"---! ERROR processing {os.path.basename(pc_path)}: {e} !---")  
        import traceback  
        traceback.print_exc()  

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

if __name__ == '__main__':  
    print(f"--- Starting Point Cloud Verification ---")  
    print(f"Reading data from: {DATASET_DIR}")  
    print(f"Saving output to: {OUTPUT_DIR}")  

    annotation_files = glob.glob(os.path.join(DATASET_DIR, 'annotations', '*.json'))  

    if not annotation_files:  
        print("ERROR: No annotation files found. Did you run the generation script first?")  
        exit()  

    N_debud = 100
    annotation_files = annotation_files[:N_debud]

    for ann_path in tqdm(annotation_files, desc="Visualizing Point Clouds"):  
        with open(ann_path, 'r') as f:  
            annotations = json.load(f)  

        pc_rel_path = annotations.get('point_cloud_path')  
        if not pc_rel_path:  
            continue  

        pc_full_path = os.path.join(DATASET_DIR, pc_rel_path)  
        if not os.path.exists(pc_full_path):  
            continue  
            
        model_name = annotations.get('model_name', os.path.splitext(os.path.basename(ann_path))[0])  
        output_filename = f"{model_name}_pointcloud.png"  
        output_path = os.path.join(OUTPUT_DIR, output_filename)  

        if not os.path.exists(output_path):  
            visualize_point_cloud(pc_full_path, output_path)  

    print("\n--- Verification complete! Check the 'verification_pointclouds' folder. ---")  
