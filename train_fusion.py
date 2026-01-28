# train.py
import torch
from torch.utils.data import DataLoader
from models.yolo_3d_fusion_model import YOLOv8_3DFusion
from utils.dataset import FusionDataset
from utils.quaternion_loss import quaternion_cosine_loss, quat_angle_error, geodesic_loss
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics import YOLO
from trainers.utils import fusion_collate_fn
import types
from torch.optim.lr_scheduler import CosineAnnealingLR
import json
import matplotlib.pyplot as plt
import os

def build_yolo_targets(batch):
    B = batch['bbox'].size(0)
    device = batch['bbox'].device
    return {
        'cls': torch.zeros(B, dtype=torch.long, device=device),
        'bboxes': batch['bbox'].clone(),
        'batch_idx': torch.arange(B, device=device)
    }

if __name__ == '__main__':
    # ========== 1. Инициализация модели ==========
    ckpt_path = 'C:/Users/walle/Desktop/nir/abc-dataset/runs/detect/yolov8n_baseline_parts5/weights/best.pt'
    print("🔍 Загружаем чекпоинт...")
    ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)

    hyp = ckpt.get('train_args', {}).get('hyp', {
        'box': 7.5, 'cls': 0.5, 'dfl': 1.5,
        'label_smoothing': 0.0, 'fl_gamma': 0.0, 'anchor_t': 4.0
    })
    args_dict = ckpt.get('train_args', {})
    if not args_dict:
        args_dict = {'nc': 80, 'imgsz': 640, 'batch': 16, 'name': 'yolov8n'}
    args_dict['nc'] = 1

    print("🔧 Создаём модель...")
    model = YOLOv8_3DFusion(cfg='yolov8n.yaml', nc=1).cuda()

    base_yolo = YOLO(ckpt_path)
    pretrained_dict = base_yolo.model.state_dict()
    model_dict = model.state_dict()
    filtered_dict = {
        k: v for k, v in pretrained_dict.items()
        if k in model_dict and v.shape == model_dict[k].shape
    }
    print(f"✅ Загружено {len(filtered_dict)} / {len(pretrained_dict)} слоёв")
    model.load_state_dict(filtered_dict, strict=False)

    model.hyp = hyp
    model.args = types.SimpleNamespace(**args_dict)
    assert hasattr(model, 'stride'), "Model missing 'stride'"
    print(f"✅ stride: {model.stride}")
    print(f"✅ hyp: {model.hyp}")

    # ========== 2. Потери и оптимизатор ==========
    detection_loss_fn = v8DetectionLoss(model, tal_topk=10)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=50)
    print("✅ Инициализация завершена")

    # ========== 3. Датасет ==========
    print("📂 Загружаем датасет...")
    train_dataset = FusionDataset("C:/Users/walle/Desktop/nir/abc-dataset/data", split="train")
    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=fusion_collate_fn,
        num_workers=0
    )

    # ========== 4. Инициализация логов ==========
    log_dir = "logs"
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "training_log.json")

    metrics_log = {
        "epochs": [],
        "total_loss": [],
        "det_loss": [],
        "quat_loss": [],
        "ang_error_deg": [],
        "grad_norm": []
    }

    # ========== 5. Обучение ==========
    print("\n🚀 Начинаем обучение...")
    for epoch in range(150):
        model.train()
        epoch_total = 0.0
        epoch_det = 0.0
        epoch_quat = 0.0
        epoch_ang = 0.0
        epoch_grad = 0.0
        num_steps = 0

        for step, batch in enumerate(train_loader):
            batch_gpu = {
                'img': batch['img'].cuda(),
                'bbox_abs': batch['bbox_abs'].cuda(),
                'quat': batch['quat'].cuda(),
                'img_hw': batch['img_hw'].cuda(),
                'bbox': batch['bbox'].cuda(),
                'pointcloud': [pc.cuda() for pc in batch['pointcloud']]
            }

            det_out, quat_pred = model(
                batch_gpu['img'],
                bbox_abs=batch_gpu['bbox_abs'],
                pointclouds=batch_gpu['pointcloud'],
                img_hw=batch_gpu['img_hw']
            )

            yolo_targets = build_yolo_targets(batch_gpu)
            loss_det, _ = detection_loss_fn(det_out, yolo_targets)
            loss_det = loss_det.sum()
            loss_quat = geodesic_loss(quat_pred, batch_gpu['quat'])
            total_loss = loss_det + 1.0 * loss_quat

            optimizer.zero_grad()
            total_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()
            scheduler.step()

            ang_error = quat_angle_error(quat_pred, batch_gpu['quat']).mean()

            # Накопление для усреднения по эпохе
            epoch_total += total_loss.item()
            epoch_det += loss_det.item()
            epoch_quat += loss_quat.item()
            epoch_ang += ang_error.item()
            epoch_grad += grad_norm.item()
            num_steps += 1

        # Усреднение по эпохе
        avg_total = epoch_total / num_steps
        avg_det = epoch_det / num_steps
        avg_quat = epoch_quat / num_steps
        avg_ang = epoch_ang / num_steps
        avg_grad = epoch_grad / num_steps

        # Логирование
        metrics_log["epochs"].append(epoch)
        metrics_log["total_loss"].append(avg_total)
        metrics_log["det_loss"].append(avg_det)
        metrics_log["quat_loss"].append(avg_quat)
        metrics_log["ang_error_deg"].append(avg_ang)
        metrics_log["grad_norm"].append(avg_grad)

        print(f"Epoch {epoch} — "
              f"Total: {avg_total:.4f}, "
              f"Det: {avg_det:.4f}, "
              f"Quat: {avg_quat:.4f}, "
              f"AngErr: {avg_ang:.1f}°, "
              f"GradNorm: {avg_grad:.2f}")
        

    torch.save(model.state_dict(), 'model_fusion.pt')

    # ========== 6. Сохранение лога ==========
    with open(log_file, 'w') as f:
        json.dump(metrics_log, f, indent=4)
    print(f"\n✅ Лог сохранён: {log_file}")

    # ========== 7. Визуализация ==========
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    epochs = metrics_log["epochs"]

    axs[0, 0].plot(epochs, metrics_log["total_loss"], label="Total Loss", color="black")
    axs[0, 0].set_title("Total Loss")
    axs[0, 0].set_xlabel("Epoch")
    axs[0, 0].grid(True)

    axs[0, 1].plot(epochs, metrics_log["det_loss"], label="Detection Loss", color="blue")
    axs[0, 1].set_title("Detection Loss")
    axs[0, 1].set_xlabel("Epoch")
    axs[0, 1].grid(True)

    axs[1, 0].plot(epochs, metrics_log["quat_loss"], label="Quaternion Loss", color="red")
    axs[1, 0].set_title("Quaternion Loss (Geodesic)")
    axs[1, 0].set_xlabel("Epoch")
    axs[1, 0].grid(True)

    axs[1, 1].plot(epochs, metrics_log["ang_error_deg"], label="Angular Error", color="purple")
    axs[1, 1].set_title("Angular Error (°)")
    axs[1, 1].set_xlabel("Epoch")
    axs[1, 1].grid(True)

    plt.tight_layout()
    plot_path = os.path.join(log_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=150)
    plt.show()
    print(f"✅ Графики сохранены: {plot_path}")

    print("\n✅ Обучение и визуализация завершены!")