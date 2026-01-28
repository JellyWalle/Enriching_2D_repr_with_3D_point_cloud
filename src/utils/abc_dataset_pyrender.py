import os
os.environ["PYOPENGL_PLATFORM"] = "osmesa"  # для pyrender
os.environ["PYGLET_HEADLESS"] = "1"        # для trimesh 
os.environ["MESA_GL_VERSION_OVERRIDE"] = "3.3"
os.environ["MESA_GLSL_VERSION_OVERRIDE"] = "330"

# --- Импорты ---
import glob
import json
import numpy as np
import trimesh
from PIL import Image, ImageDraw
from tqdm import tqdm
import io
import traceback
from scipy.spatial.transform import Rotation as R

# --- Константы ---
IMG_SIZE = (512, 512)
FOV_DEG = 60.0
N_VIEWS = 12

# --- Пути ---
ROOT_DIR = "/mnt/hdd_works/ml_640/yolov11ir/abc_full_10k_v00/10k"
INPUT_DATA_DIR = os.path.join(ROOT_DIR, "train", "512")
OUTPUT_DIR = os.path.join(ROOT_DIR, "dataset_generated_final_pyrender")

# --- Количество моделей ---
NUM_MODELS_TO_PROCESS = 8000  # или -1 для всех

# --- Создание директорий ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
for subdir in ['views', 'points', 'annotations', 'views_debug']:
    os.makedirs(os.path.join(OUTPUT_DIR, subdir), exist_ok=True)


# --- ФУНКЦИИ ---
def load_and_normalize_mesh(obj_path):
    try:
        mesh = trimesh.load(obj_path, force='mesh', process=False)
        if isinstance(mesh, trimesh.Scene):
            if not mesh.geometry:
                return None
            mesh = trimesh.util.concatenate(mesh.geometry.values())
        if not hasattr(mesh, 'vertices') or mesh.vertices.size == 0:
            return None

        mesh.fix_normals()
        mesh.vertices -= mesh.bounds.mean(axis=0)
        max_extent = np.max(mesh.extents)
        if max_extent < 1e-6:
            return None
        mesh.vertices /= max_extent
        return mesh
    except Exception:
        return None


def get_random_camera_positions(n_views, radius, min_height_ratio=0.1, max_height_ratio=0.5):
    positions = []
    for _ in range(n_views):
        theta = np.random.uniform(0, 2 * np.pi)
        phi_min_rad = np.arccos(max_height_ratio)
        phi_max_rad = np.arccos(min_height_ratio)
        phi = np.random.uniform(phi_min_rad, phi_max_rad)
        x = radius * np.sin(phi) * np.cos(theta)
        y = radius * np.cos(phi)
        z = radius * np.sin(phi) * np.sin(theta)
        random_radius = radius * np.random.uniform(0.9, 1.1)
        positions.append(np.array([x, y, z]) * (random_radius / radius))
    return positions


def get_random_up_vector(view_direction):
    if np.abs(np.dot(view_direction, [0, 1, 0])) > 0.99:
        v_aux = np.array([1, 0, 0])
    else:
        v_aux = np.array([0, 1, 0])
    right_vector = np.cross(view_direction, v_aux)
    right_vector /= np.linalg.norm(right_vector)
    standard_up = np.cross(right_vector, view_direction)
    standard_up /= np.linalg.norm(standard_up)
    random_roll_angle = np.random.uniform(0, 2 * np.pi)
    K = np.array([
        [0, -view_direction[2], view_direction[1]],
        [view_direction[2], 0, -view_direction[0]],
        [-view_direction[1], view_direction[0], 0]
    ])
    R_mat = np.eye(3) + np.sin(random_roll_angle) * K + (1 - np.cos(random_roll_angle)) * (K @ K)
    random_up = R_mat @ standard_up
    return random_up


def look_at_matrix(eye, target, up):
    eye, target, up = np.asarray(eye), np.asarray(target), np.asarray(up)
    f = target - eye
    f_norm = np.linalg.norm(f)
    if f_norm < 1e-6:
        return np.eye(4)
    f /= f_norm
    s = np.cross(f, up)
    s_norm = np.linalg.norm(s)
    if s_norm < 1e-6:
        return np.eye(4)
    s /= s_norm
    u = np.cross(s, f)
    view_matrix = np.eye(4)
    view_matrix[0, :3] = s
    view_matrix[1, :3] = u
    view_matrix[2, :3] = -f
    view_matrix[:3, 3] = -np.dot(view_matrix[:3, :3], eye)
    return view_matrix


def matrix_to_quaternion(view_matrix):
    rotation_matrix = view_matrix[:3, :3]
    r = R.from_matrix(rotation_matrix)
    quat_xyzw = r.as_quat()
    return [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]]


# ---  РЕНДЕРИНГ С PYRENDER ---
import pyrender

def render_mesh_pyrender(mesh, cam_pos, up_vector, img_size=(512, 512), fov_deg=60.0):
    # Добавляем ambient light + directional light
    scene = pyrender.Scene(
        bg_color=np.array([0.0, 0.0, 0.0, 0.0]),
        ambient_light=np.array([0.5, 0.5, 0.5, 1.0])
    )
    mesh_pr = pyrender.Mesh.from_trimesh(mesh, smooth=False)
    scene.add(mesh_pr)

    #  добавим направленный свет
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=2.0)
    light_pose = np.eye(4)
    light_pose[:3, 3] = [0, 0, 3]  # свет сверху
    scene.add(light, pose=light_pose)

    view_matrix_wc = look_at_matrix(eye=cam_pos, target=[0, 0, 0], up=up_vector)
    cam_pose = np.linalg.inv(view_matrix_wc)
    camera = pyrender.PerspectiveCamera(yfov=np.deg2rad(fov_deg), aspectRatio=img_size[0] / img_size[1])
    scene.add(camera, pose=cam_pose)

    renderer = pyrender.OffscreenRenderer(viewport_width=img_size[0], viewport_height=img_size[1])
    try:
        color, _ = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()

    if color.shape[2] == 4:
        color = color[:, :, :3]
    return color.astype(np.uint8)

def project_points_to_image_pil(img_array, scene_camera_params, points_3d):
    """
    Альтернативный подход: используем те же матрицы, но не зависим от trimesh.Scene.
    Передаём параметры камеры явно.
    """
    cam_pos, target, up, fov_deg, img_size = scene_camera_params
    width, height = img_size

    # Матрица вида (world-to-camera)
    view_matrix = look_at_matrix(eye=cam_pos, target=target, up=up)
    
    # Проекционная матрица
    fov_y = np.deg2rad(fov_deg)
    aspect_ratio = width / height
    near_clip = 0.01
    far_clip = 100.0
    f = 1.0 / np.tan(fov_y / 2.0)
    projection_matrix = np.array([
        [f / aspect_ratio, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far_clip + near_clip) / (near_clip - far_clip), (2 * far_clip * near_clip) / (near_clip - far_clip)],
        [0, 0, -1, 0]
    ])

    mvp = projection_matrix @ view_matrix
    points_3d_h = np.hstack((points_3d, np.ones((len(points_3d), 1))))
    points_clip = (mvp @ points_3d_h.T).T
    w = points_clip[:, 3]
    valid = w > 1e-4
    if not np.any(valid):
        return None, None
    points_ndc = points_clip[valid, :3] / w[valid, np.newaxis]
    in_screen = (np.abs(points_ndc[:, 0]) < 1.0) & (np.abs(points_ndc[:, 1]) < 1.0)
    if not np.any(in_screen):
        return None, None
    points_ndc = points_ndc[in_screen]
    points_2d = np.zeros((len(points_ndc), 2))
    points_2d[:, 0] = (points_ndc[:, 0] + 1.0) * 0.5 * width
    points_2d[:, 1] = (1.0 - (points_ndc[:, 1] + 1.0) * 0.5) * height
    x_min, y_min = points_2d.min(axis=0)
    x_max, y_max = points_2d.max(axis=0)
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(width - 1, x_max)
    y_max = min(height - 1, y_max)
    if x_max <= x_min or y_max <= y_min:
        return points_2d, None
    bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
    return points_2d, bbox


def process_one_model(obj_path, n_views):
    model_name = os.path.splitext(os.path.basename(obj_path))[0]
    mesh = load_and_normalize_mesh(obj_path)
    if mesh is None:
        return

    bounding_radius = np.max(np.linalg.norm(mesh.vertices, axis=1))
    optimal_radius = bounding_radius / np.tan(np.deg2rad(FOV_DEG / 2.0)) * 1.8
    pc_path = os.path.join(OUTPUT_DIR, 'points', f'{model_name}.npy')
    np.save(pc_path, mesh.vertices)

    cam_positions = get_random_camera_positions(n_views, radius=optimal_radius)
    annotations = {
        'model_name': model_name,
        'point_cloud_path': os.path.relpath(pc_path, OUTPUT_DIR),
        'views': []
    }

    for i, cam_pos in enumerate(cam_positions):
        try:
            view_name = f'view_{i:03d}'
            target_pos = np.array([0, 0, 0])
            view_direction = target_pos - cam_pos
            view_direction /= np.linalg.norm(view_direction)
            up_vector = get_random_up_vector(view_direction)

            # Рендеринг через pyrender
            img_array = render_mesh_pyrender(
                mesh,
                cam_pos=cam_pos,
                up_vector=up_vector,
                img_size=IMG_SIZE,
                fov_deg=FOV_DEG
            )
            img_pil = Image.fromarray(img_array, mode='RGB')

            if np.max(img_array) < 10:
                print(f"{view_name}: empty frame")
                continue

            # Проекция точек (без trimesh.Scene)
            scene_params = (cam_pos, target_pos, up_vector, FOV_DEG, IMG_SIZE)
            points_2d, bbox_2d = project_points_to_image_pil(img_array, scene_params, mesh.vertices)
            if bbox_2d is None:
                print(f"{view_name}: bbox is none")
                continue

            # Сохранение изображений
            img_path = os.path.join(OUTPUT_DIR, 'views', f'{model_name}_{view_name}.png')
            img_pil.save(img_path)

            img_debug_pil = img_pil.copy()
            draw = ImageDraw.Draw(img_debug_pil)
            draw.rectangle(bbox_2d, outline="green", width=1)
            if points_2d is not None and len(points_2d) > 0:
                draw.point([tuple(p) for p in points_2d], fill="red")
            img_debug_path = os.path.join(OUTPUT_DIR, 'views_debug', f'{model_name}_{view_name}_debug.png')
            img_debug_pil.save(img_debug_path)

            orientation_quaternion = matrix_to_quaternion(look_at_matrix(cam_pos, target_pos, up_vector))
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

        except Exception as e:
            print(f"\n  - Warning: Skipping view {i} for model {model_name} due to error: {e}")
            continue

    if annotations['views']:
        ann_path = os.path.join(OUTPUT_DIR, 'annotations', f'{model_name}.json')
        with open(ann_path, 'w') as f:
            json.dump(annotations, f, indent=2)


# --- MAIN ---
if __name__ == '__main__':
    print("--- Starting Robust Dataset Generation (pyrender) ---")
    obj_files = glob.glob(os.path.join(INPUT_DATA_DIR, '**', '*.obj'), recursive=True)
    
    if NUM_MODELS_TO_PROCESS > 0:
        obj_files_subset = obj_files[:NUM_MODELS_TO_PROCESS]
    else:
        obj_files_subset = obj_files

    print(f"Found {len(obj_files)} models. Processing {len(obj_files_subset)}.")
    print(f"Output will be saved to: {OUTPUT_DIR}")

    for obj_path in tqdm(obj_files_subset, desc="Processing models"):
        try:
            process_one_model(obj_path, n_views=N_VIEWS)
        except Exception as e:
            print(f"\n---!!! UNEXPECTED FATAL Error on {os.path.basename(obj_path)}: {e} !!!---")
            traceback.print_exc()

    print("\n--- Dataset generation complete. ---")