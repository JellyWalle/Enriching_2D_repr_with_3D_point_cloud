#!/usr/bin/env python3
"""
Фаза 2: Обучение ToleranceClassifier.
Использует замороженную GNN как экстрактор признаков и обучает MLP 
для бинарной классификации IN_TOL / OUT_OF_TOL.
"""
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset
from pathlib import Path
from tqdm import tqdm
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from torch_geometric.data import HeteroData
from src.models.hybrid_gnn import HybridGNN

# ==============================================================================
# 1. АРХИТЕКТУРА КЛАССИФИКАТОРА
# ==============================================================================
class ToleranceClassifier(nn.Module):
    def __init__(self, input_dim=397):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)

# ==============================================================================
# 2. УТИЛИТЫ ДЛЯ ЗАГРУЗКИ ДАННЫХ (КОПИЯ ИЗ TRAIN.PY)
# ==============================================================================
def clean_json_keys(obj):
    """Рекурсивно удаляет пробелы в концах ключей и значений строк."""
    if isinstance(obj, dict):
        return {k.strip(): clean_json_keys(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_keys(item) for item in obj]
    elif isinstance(obj, str):
        return obj.strip()
    return obj

TYPE_TO_ID = {
    'plane': 0, 'cylinder': 1, 'through_hole': 2, 'blind_hole': 2,
    'pocket': 3, 'slot': 4, 'cone': 5, 'sphere': 6, 'chamfer': 7, 'unknown': 8
}

def normalize_3d_features_on_the_fly(node_data: dict) -> list:
    """Применяет логарифмическую нормализацию и извлекает топологические признаки."""
    measured_value = node_data.get('measured_value', {})
    geometry = node_data.get('geometry', {})
    topology = node_data.get('topology', {})
    confidence = float(node_data.get('confidence', 0.9))
    ftype = node_data.get('type', 'unknown')
    
    features = [0.0] * 24
    features[0] = float(TYPE_TO_ID.get(ftype, 8)) / 10.0
    
    def safe_log(val):
        try: return math.log1p(max(0.0, float(val)))
        except: return 0.0
            
    if 'radius' in measured_value:
        features[1] = safe_log(measured_value['radius'])
        features[2] = safe_log(float(measured_value['radius']) * 2.0)
    if 'diameter' in measured_value:
        features[2] = safe_log(measured_value['diameter'])
        features[1] = safe_log(float(measured_value['diameter']) / 2.0)
    if 'length' in measured_value: features[3] = safe_log(measured_value['length'])
    if 'width' in measured_value: features[4] = safe_log(measured_value['width'])
    if 'depth' in measured_value: features[5] = safe_log(measured_value['depth'])
    if 'height' in measured_value: features[6] = safe_log(measured_value['height'])
    if 'base_radius' in measured_value: features[7] = safe_log(measured_value['base_radius'])
    if 'top_radius' in measured_value: features[8] = safe_log(measured_value['top_radius'])
    
    area = abs(float(geometry.get('surface_area_mm2', 1.0)))
    features[9] = math.log1p(area)
    
    curv = geometry.get('curvature', {})
    features[10] = float(curv.get('mean', 0.0)) * 10.0
    features[11] = float(curv.get('std', 0.0)) * 10.0
    features[12] = float(curv.get('max', 0.0)) * 10.0
    features[13] = float(geometry.get('normal_std', 0.0))
    
    adj_segments = topology.get('adjacent_segments', [])
    features[14] = float(sum(1 for r in adj_segments if r.get('relation') == 'convex')) / 10.0
    features[15] = float(sum(1 for r in adj_segments if r.get('relation') == 'concave')) / 10.0
    features[16] = float(sum(1 for r in adj_segments if r.get('relation') == 'orthogonal')) / 10.0
    
    features[17] = 1.0 if len(topology.get('contains', [])) > 0 else 0.0
    features[18] = 1.0 if len(topology.get('inside', [])) > 0 else 0.0
    features[19] = confidence
    return features

def load_graph_for_phase2(json_path: str):
    """
    Загружает граф и возвращает HeteroData, а также маппинги ID узлов в индексы тензоров.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = clean_json_keys(json.load(f))
        
    nodes_2d = data.get('nodes_2d', {})
    nodes_3d = data.get('nodes_3d', {})
    edges = data.get('edges', [])
    
    if not nodes_2d or not nodes_3d:
        return None, None, None, {}, {}
        
    n2d_ids = list(nodes_2d.keys())
    n3d_ids = list(nodes_3d.keys())
    id2d_to_idx = {nid: i for i, nid in enumerate(n2d_ids)}
    id3d_to_idx = {nid: i for i, nid in enumerate(n3d_ids)}
    
    x_2d_list = []
    for nid in n2d_ids:
        node = nodes_2d[nid]
        val = float(node.get('value', 0.0))
        tol = float(node.get('tolerance', {}).get('value', 0.0))
        pos = node.get('position_2d', [0.0, 0.0])
        # Если вы удаляли text_emb в train.py, уберите эту строку и 768 нулей
        text_emb = node.get('text_embedding', [0.0]*768) 
        visual_emb = node.get('visual_features', [0.0]*256)
        x_2d_list.append([val, tol, pos[0], pos[1]]  + visual_emb)
        
    x_3d_list = []
    for nid in n3d_ids:
        node = nodes_3d[nid]
        features = normalize_3d_features_on_the_fly(node)
        x_3d_list.append(features)
        
    hetero_data = HeteroData()
    hetero_data['2d'].x = torch.tensor(x_2d_list, dtype=torch.float32)
    hetero_data['3d'].x = torch.tensor(x_3d_list, dtype=torch.float32)
    
    edge_dict = {}
    for edge in edges:
        src, dst, rel = edge['src'], edge['dst'], edge['type']
        if rel == 'adjacent_2d':
            key = ('2d', 'adjacent_2d', '2d')
            if key not in edge_dict: edge_dict[key] = [[], []]
            if src in id2d_to_idx and dst in id2d_to_idx:
                edge_dict[key][0].append(id2d_to_idx[src])
                edge_dict[key][1].append(id2d_to_idx[dst])
        elif rel.startswith('adjacent_3d_'):
            key = ('3d', rel, '3d')
            if key not in edge_dict: edge_dict[key] = [[], []]
            if src in id3d_to_idx and dst in id3d_to_idx:
                edge_dict[key][0].append(id3d_to_idx[src])
                edge_dict[key][1].append(id3d_to_idx[dst])
        elif rel == 'correspondence':
            # Для Фазы 2 нам важны ВСЕ ребра, чтобы MLP научился отличать IN от OUT
            weight = float(edge.get('weight', 0.0))
            if weight >= 0.50: 
                key = ('2d', 'correspondence', '3d')
                if key not in edge_dict: edge_dict[key] = [[], []]
                if src in id2d_to_idx and dst in id3d_to_idx:
                    edge_dict[key][0].append(id2d_to_idx[src])
                    edge_dict[key][1].append(id3d_to_idx[dst])
                    
    for (src_type, rel_type, dst_type), indices in edge_dict.items():
        if len(indices[0]) > 0:
            hetero_data[src_type, rel_type, dst_type].edge_index = torch.tensor(indices, dtype=torch.long)
            
    return hetero_data, nodes_2d, nodes_3d, id2d_to_idx, id3d_to_idx

# ==============================================================================
# 3. ИЗВЛЕЧЕНИЕ ПРИЗНАКОВ
# ==============================================================================
NUM_TYPES = 9

def get_measured_value(node_3d: dict) -> float:
    """Извлекает основной размер из 3D узла."""
    mv = node_3d.get('measured_value', {})
    if not isinstance(mv, dict): return 0.0
    # Приоритет: diameter, затем radius*2, затем остальные
    if 'diameter' in mv: return float(mv['diameter'])
    if 'radius' in mv: return float(mv['radius']) * 2.0
    for k in ['length', 'width', 'height', 'max_dimension']:
        if k in mv: return float(mv[k])
    return 0.0

def extract_pair_features(z_2d_vec, z_3d_vec, node_2d: dict, node_3d: dict):
    """Формирует вектор признаков (397D) для пары."""
    # ВАЖНО: Определяем устройство, на котором находятся эмбеддинги
    device = z_2d_vec.device
    
    # 1. Эмбеддинги и их разность (128 * 3 = 384)
    diff = torch.abs(z_2d_vec - z_3d_vec)
    
    # 2. Косинусное сходство (1)
    cos_sim = F.cosine_similarity(z_2d_vec.unsqueeze(0), z_3d_vec.unsqueeze(0)).item()
    
    # 3. Геометрические признаки (3)
    nominal = float(node_2d.get('value', 0.0))
    tol_info = node_2d.get('tolerance', {})
    tol_val = float(tol_info.get('value', nominal * 0.05)) if isinstance(tol_info, dict) else float(tol_info)
    if tol_val == 0 and nominal > 0: tol_val = nominal * 0.05
    
    measured = get_measured_value(node_3d)
    deviation = abs(nominal - measured)
    relative_error = deviation / (tol_val + 1e-6)
    
    # Логарифмируем геометрические признаки, чтобы они не доминировали над эмбеддингами
    log_dev = math.log1p(deviation)
    log_tol = math.log1p(tol_val)
    log_rel_err = math.log1p(relative_error)
    
    # 4. One-hot для типа признака (9)
    text = node_2d.get('text_content', '').lower()
    if '⌀' in text or 'diameter' in text: type_id = 2 # hole
    elif 'width' in text or 'slot' in text: type_id = 4 # slot
    else: type_id = 0 # plane/generic
    
    one_hot = [0.0] * NUM_TYPES
    one_hot[type_id] = 1.0
    
    # === ИСПРАВЛЕНИЕ: Явно указываем device для новых тензоров ===
    scalar_features = torch.tensor(
        [cos_sim, log_dev, log_tol, log_rel_err] + one_hot, 
        device=device,  # <-- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ
        dtype=torch.float32
    )
    
    return torch.cat([
        z_2d_vec,      # 128 (уже на GPU)
        z_3d_vec,      # 128 (уже на GPU)
        diff,          # 128 (уже на GPU)
        scalar_features # 14 (теперь тоже на GPU)
    ])
# ==============================================================================
# 4. СБОР ДАТАСЕТА
# ==============================================================================
def collect_phase2_dataset(gnn_model, data_dir: str, device: str):
    X_list, Y_list = [], []
    gnn_model.eval()
    
    part_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
    print(f"🔍 Найдено {len(part_dirs)} папок. Начинаем извлечение признаков...")
    
    skipped = 0
    for p_dir in tqdm(part_dirs, desc="Phase 2 Data Collection"):
        for fname in ["graph_in_tol.json", "graph_out_tol.json"]:
            g_path = p_dir / "graphs" / fname
            if not g_path.exists(): continue
            
            try:
                data, n2d_dict, n3d_dict, map2d, map3d = load_graph_for_phase2(str(g_path))
                if data is None or ('2d', 'correspondence', '3d') not in data.edge_index_dict:
                    skipped += 1
                    continue
                
                data = data.to(device)
                with torch.no_grad():
                    z_2d, z_3d = gnn_model(data.x_dict, data.edge_index_dict)
                
                corr_edges = data.edge_index_dict[('2d', 'correspondence', '3d')]
                
                # Нам нужно пройтись по оригинальным JSON ребрам, чтобы взять tolerance_status
                with open(g_path, 'r', encoding='utf-8') as f:
                    raw_json = clean_json_keys(json.load(f))
                
                # Создаем быстрый доступ к JSON ребрам
                json_edges_map = {}
                for e in raw_json.get('edges', []):
                    if e.get('type') == 'correspondence':
                        json_edges_map[(e['src'], e['dst'])] = e
                
                # Итерируемся по тензорным ребрам (они уже отфильтрованы по весу >= 0.5)
                for i in range(corr_edges.shape[1]):
                    idx_2d = corr_edges[0, i].item()
                    idx_3d = corr_edges[1, i].item()
                    
                    # Находим оригинальные ID
                    src_id = next((k for k, v in map2d.items() if v == idx_2d), None)
                    dst_id = next((k for k, v in map3d.items() if v == idx_3d), None)
                    
                    if not src_id or not dst_id: continue
                    if (src_id, dst_id) not in json_edges_map: continue
                    
                    json_edge = json_edges_map[(src_id, dst_id)]
                    status = json_edge.get('features', {}).get('tolerance_status', 'UNKNOWN')
                    if status == 'UNKNOWN': continue
                    
                    label = 1.0 if status == 'IN_TOLERANCE' else 0.0
                    
                    n2d = n2d_dict[src_id]
                    n3d = n3d_dict[dst_id]
                    
                    feat_vec = extract_pair_features(z_2d[idx_2d], z_3d[idx_3d], n2d, n3d)
                    X_list.append(feat_vec.cpu())
                    Y_list.append(label)
                    
            except Exception as e:
                print(f"    Error processing {g_path}: {e}")
                continue

    print(f"  Собрано {len(X_list)} примеров. Пропущено файлов: {skipped}")
    return torch.stack(X_list), torch.tensor(Y_list, dtype=torch.float32)

# ==============================================================================
# 5. ГЛАВНЫЙ ЦИКЛ
# ==============================================================================
def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"🚀 Запуск Фазы 2 на устройстве: {device}")
    
    # 1. Загрузка и заморозка GNN
    print("🧠 Загрузка обученной GNN...")
    # Определяем размерности динамически, чтобы избежать ошибок 1028 vs 260
    sample_path = next(Path("data/processed_full").glob("*/graphs/graph_in_tol.json"))
    sample_data, _, _, _, _ = load_graph_for_phase2(str(sample_path))
    dim_2d = sample_data['2d'].x.shape[1]
    dim_3d = sample_data['3d'].x.shape[1]
    print(f"   Размерности: 2D={dim_2d}, 3D={dim_3d}")
    
    gnn = HybridGNN(input_dim_2d=dim_2d, input_dim_3d=dim_3d, hidden_dim=64).to(device)
    
    ckpt_path = "checkpoints/best_model.pth" # Или best_model.pth
    if Path(ckpt_path).exists():
        checkpoint = torch.load(ckpt_path, map_location=device)
        gnn.load_state_dict(checkpoint['model_state_dict'])
        print(f"     Загружены веса из {ckpt_path}")
    else:
        print("       Чекпоинт не найден! Используем случайные веса GNN (для теста архитектуры).")
        
    for param in gnn.parameters():
        param.requires_grad = False
    gnn.eval()
    
    # 2. Сбор датасета
    print("\n📊 Извлечение признаков из графов...")
    X, Y = collect_phase2_dataset(gnn, "data/processed_full", device)
    
    if len(X) == 0:
        print("   Не удалось собрать данные. Завершение.")
        return
        
    # 3. Разделение на Train/Val
    X_train, X_val, Y_train, Y_val = train_test_split(X, Y, test_size=0.2, random_state=42, stratify=Y)
    
    train_dataset = TensorDataset(X_train, Y_train)
    val_dataset = TensorDataset(X_val, Y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=512, shuffle=False)
    
    print(f"📦 Train: {len(X_train)} | Val: {len(X_val)}")
    print(f"🎯 Баланс классов (Train): IN={Y_train.sum().item()}, OUT={(1-Y_train).sum().item()}")
    
    # 4. Инициализация MLP
    input_dim = X_train.shape[1]
    print(f"\n🏗️ Инициализация ToleranceClassifier (input_dim={input_dim})...")
    mlp = ToleranceClassifier(input_dim=input_dim).to(device)
    optimizer = optim.AdamW(mlp.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    criterion = nn.BCEWithLogitsLoss()
    
    # 5. Обучение
    print("\n🏋️ Обучение MLP...")
    best_f1 = 0.0
    Path("checkpoints").mkdir(exist_ok=True)
    
    for epoch in range(1, 10):
        mlp.train()
        train_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = mlp(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        # Валидация
        mlp.eval()
        val_preds, val_true = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                logits = mlp(xb)
                probs = torch.sigmoid(logits).cpu().numpy()
                val_preds.extend((probs > 0.5).astype(float))
                val_true.extend(yb.numpy())
                
        val_f1 = f1_score(val_true, val_preds, zero_division=0)
        val_p = precision_score(val_true, val_preds, zero_division=0)
        val_r = recall_score(val_true, val_preds, zero_division=0)
        
        scheduler.step(val_f1)
        current_lr = optimizer.param_groups[0]['lr']
        
        print(f"Epoch {epoch:02d} | Loss: {train_loss/len(train_loader):.4f} | "
              f"F1: {val_f1:.4f} | P: {val_p:.4f} | R: {val_r:.4f} | LR: {current_lr:.1e}")
        
        if val_f1 > best_f1:
            best_f1 = val_f1
            torch.save(mlp.state_dict(), "checkpoints/tolerance_classifier.pth")
            print(f"  💾 Сохранена лучшая модель (F1: {best_f1:.4f})")
            
    print("\n  Обучение Фазы 2 завершено!")
    print(f"🏆 Лучший F1-score: {best_f1:.4f}")
    print(f"  Модель сохранена в: checkpoints/tolerance_classifier.pth")

if __name__ == '__main__':
    main()