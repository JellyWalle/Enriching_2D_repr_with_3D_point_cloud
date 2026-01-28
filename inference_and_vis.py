import torch
import cv2
import numpy as np
import os
from yolo_3d_fusion_model import YOLOv8_3DFusion
from ultralytics.utils.plotting import Annotator
from PIL import Image
from scipy.spatial.transform import Rotation as R

# -----------------------------
# Конфигурация
# -----------------------------
CKPT_PATH = "model_fusion.pt"
IMAGE_PATH = "dataset_generated_final_pyrender/views/0000101_partstudio_07_model_ste_00_512_view_000.png"
PC_PATH = "dataset_generated_final_pyrender/points/0000101_partstudio_07_model_ste_00_512.npy"
BBOX_ABS = [132, 194, 402, 300]  # GT bbox (используется для ROI)
IMG_SIZE = (512, 512)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Параметры камеры — должны совпадать с теми, что использовались при генерации!
FOV_DEG = 60.0
AXIS_LENGTH = 0.9  # длина осей в нормализованных координатах

# -----------------------------
# Вспомогательные функции для проекции осей
# -----------------------------

def quaternion_to_rotation_matrix(quat_wxyz):
    """Конвертирует [w, x, y, z] → матрица 3x3"""
    quat_xyzw = [quat_wxyz[1], quat_wxyz[2], quat_wxyz[3], quat_wxyz[0]]
    return R.from_quat(quat_xyzw).as_matrix()

def build_projection_matrix(fov_deg, width, height):
    fov_y_rad = np.deg2rad(fov_deg)
    aspect = width / height
    near, far = 0.01, 100.0
    f = 1.0 / np.tan(fov_y_rad / 2.0)
    return np.array([
        [f / aspect, 0, 0, 0],
        [0, f, 0, 0],
        [0, 0, (far + near) / (near - far), (2 * far * near) / (near - far)],
        [0, 0, -1, 0]
    ])

def project_3d_to_2d(points_3d, view_matrix, proj_matrix, img_w, img_h):
    """Проекция 3D → 2D (экран)"""
    points_h = np.hstack([points_3d, np.ones((len(points_3d), 1))])  # [N, 4]
    points_clip = (proj_matrix @ view_matrix @ points_h.T).T  # [N, 4]
    w = points_clip[:, 3]
    valid = w > 1e-5
    if not np.any(valid):
        return None
    ndc = points_clip[valid, :3] / w[valid, None]
    pts_2d = np.zeros_like(ndc[:, :2])
    pts_2d[:, 0] = (ndc[:, 0] + 1) * 0.5 * img_w
    pts_2d[:, 1] = (1 - (ndc[:, 1] + 1) * 0.5) * img_h
    return pts_2d.astype(int)

def draw_axes(image, origin, axes_ends):
    """Рисует X (красный), Y (зелёный), Z (синий)"""
    colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # BGR
    for end, color in zip(axes_ends, colors):
        cv2.line(image, tuple(origin), tuple(end), color, 2)
    return image

# -----------------------------
# Загрузка модели
# -----------------------------
print("🔧 Загружаем модель...")
model = YOLOv8_3DFusion(cfg='yolov8n.yaml', nc=1).to(DEVICE)
model.load_state_dict(torch.load(CKPT_PATH, map_location=DEVICE))
model.eval()
print("✅ Модель загружена.")

# -----------------------------
# Загрузка данных
# -----------------------------
img_orig = Image.open(IMAGE_PATH).convert("RGB")
orig_h, orig_w = img_orig.height, img_orig.width
img_hw = torch.tensor([[orig_h, orig_w]], dtype=torch.float32).to(DEVICE)
img = img_orig.resize(IMG_SIZE)
img_tensor = torch.from_numpy(np.array(img)).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)

pointcloud = torch.from_numpy(np.load(PC_PATH)).float().to(DEVICE)
pointclouds = [pointcloud]
bbox_abs = torch.tensor([BBOX_ABS], dtype=torch.float32).to(DEVICE)

# -----------------------------
# Инференс
# -----------------------------
with torch.no_grad():
    det_out, quat_pred = model(
        img_tensor,
        bbox_abs=bbox_abs,
        pointclouds=pointclouds,
        img_hw=img_hw
    )
    quat_pred = quat_pred[0].cpu().numpy()  # [w, x, y, z]

# -----------------------------
# Постобработка детекции
# -----------------------------
from ultralytics.utils.ops import non_max_suppression

if isinstance(det_out, (list, tuple)):
    pred_combined = det_out
else:
    pred_combined = det_out

preds = non_max_suppression(pred_combined, conf_thres=0.25, iou_thres=0.45, max_det=100)[0]

# -----------------------------
# Визуализация: bounding box + текст + оси координат
# -----------------------------
img_vis = cv2.cvtColor(np.array(img_orig), cv2.COLOR_RGB2BGR)

# 1. Рисуем детекцию
if preds is not None and len(preds) > 0:
    x1, y1, x2, y2, conf, cls = preds[0].cpu().numpy()
    cv2.rectangle(img_vis, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
    cv2.putText(img_vis, f"part {conf:.2f}", (int(x1), int(y1) - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # 2. Конвертируем кватернион в углы Эйлера
    euler_deg = R.from_quat(quat_pred[[1, 2, 3, 0]]).as_euler('zyx', degrees=True)
    yaw, pitch, roll = euler_deg
    euler_str = f"ypr: {yaw:5.1f} {pitch:5.1f} {roll:5.1f} [deg]"
    cv2.putText(img_vis, euler_str, (int(x1), int(y1) - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

# 3. Рисуем систему координат на основе ПРЕДСКАЗАННОГО кватерниона
#    Мы предполагаем, что центр объекта — в центре bbox (приближение!)
if preds is not None and len(preds) > 0:
    # Центр объекта в 2D (приближение)
    center_2d = np.array([(x1 + x2) / 2, (y1 + y2) / 2])

    # Но для точной проекции нужно знать позицию камеры → её нет в инференсе!
    # Поэтому используем УПРОЩЁННЫЙ подход: отрисовка в центре bbox с масштабом
    # ВАЖНО: это приближение! Для точной проверки нужна позиция камеры.

    # Альтернатива: если вы знаете, что объект всегда в центре сцены (как в вашем датасете),
    # и камера смотрит на (0,0,0), то можно восстановить view_matrix из кватерниона.

    # В вашем случае — объект центрирован в (0,0,0), и кватернион описывает поворот камеры.
    # Значит, поворот объекта — обратный: R_obj = R_cam.T

    R_cam = quaternion_to_rotation_matrix(quat_pred)  # камера → мир
    R_obj = R_cam.T  # мир → объект (т.е. поворот объекта)

    # Определяем 3D точки осей (в системе объекта)
    origin_3d = np.array([[0, 0, 0]], dtype=np.float32)
    axes_3d = np.array([
        [AXIS_LENGTH, 0, 0],
        [0, AXIS_LENGTH, 0],
        [0, 0, AXIS_LENGTH]
    ], dtype=np.float32)

    # Применяем поворот объекта
    axes_3d_rot = (R_obj @ axes_3d.T).T  # [3, 3]
    origin_3d_rot = (R_obj @ origin_3d.T).T

    # Теперь нужно спроецировать. Но у нас нет позиции камеры!
    # В вашем датасете: камера смотрит на (0,0,0), и объект в (0,0,0).
    # Поэтому можно предположить, что камера находится на оси Z на расстоянии `d`.

    # Восстановим приблизительную позицию камеры по FOV и размеру bbox
    # Упрощение: используем центр изображения как проекцию (0,0,0)
    center_x, center_y = orig_w / 2, orig_h / 2

    # Определим направление взгляда: из кватерниона
    # Единичный вектор Z камеры в мировых координатах
    cam_z = R_cam @ np.array([0, 0, 1])  # направление взгляда

    # Но для проекции проще использовать ортогональную проекцию в центре bbox
    # → просто масштабируем 3D оси до 2D с учётом масштаба bbox

    bbox_w = x2 - x1
    bbox_h = y2 - y1
    scale = min(bbox_w, bbox_h) * 0.3  # длина оси в пикселях

    # Отображаем 3D оси как 2D векторы (упрощённо)
    axes_2d = []
    for axis in axes_3d_rot:
        # Берём x и y компоненты как направление на изображении
        dx = axis[0] * scale
        dy = -axis[1] * scale  # y вверх в 3D, но вниз в изображении
        axes_2d.append((int(center_2d[0] + dx), int(center_2d[1] + dy)))

    origin_2d = (int(center_2d[0]), int(center_2d[1]))
    img_vis = draw_axes(img_vis, origin_2d, axes_2d)

else:
    # Если нет детекции — просто текст с углами
    euler_deg = R.from_quat(quat_pred[[1, 2, 3, 0]]).as_euler('zyx', degrees=True)
    yaw, pitch, roll = euler_deg
    cv2.putText(img_vis, f"y={yaw:.1f} p={pitch:.1 f} r={roll:.1f}", (20, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

# -----------------------------
# Сохранение
# -----------------------------
cv2.imwrite("Prediction_with_axes.png", img_vis)
print("✅ Результат с осями сохранён как 'Prediction_with_axes.png'")