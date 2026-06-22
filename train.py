#!/usr/bin/env python3
"""
Скрипт обучения HybridGNN с контрастной потерей NT-Xent.
Основная метрика оптимизации: Recall@1 (доля правильно сопоставленных пар).
"""
import os
import json
import logging
import argparse
import math
from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.data import HeteroData
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import HeteroConv, SAGEConv
from tqdm import tqdm
from src.models.hybrid_gnn import HybridGNN, HeteroConv, NTXentLoss

# ==============================================================================
# 1. КОНФИГУРАЦИЯ И ЛОГИРОВАНИЕ
# ==============================================================================
@dataclass
class TrainConfig:
    data_dir: str = "data/preprocessed_full"
    log_dir: str = "runs/hybrid_gnn_training"
    checkpoint_dir: str = "checkpoints"
    
    # Архитектура
    hidden_dim: int = 128
    num_layers: int = 2
    
    # Обучение
    batch_size: int = 8
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 1e-3
    temperature: float = 0.2
    
    # Системные
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    num_workers: int = 4
    val_split: float = 0.2

def setup_logging(log_dir: str):
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(Path(log_dir) / "training.log"),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# ==============================================================================
# 2. DATASET И ЗАГРУЗКА ДАННЫХ
# ==============================================================================
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
        try:
            return math.log1p(max(0.0, float(val)))
        except (ValueError, TypeError):
            return 0.0
            
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
    n_convex = sum(1 for rel in adj_segments if rel.get('relation') == 'convex')
    n_concave = sum(1 for rel in adj_segments if rel.get('relation') == 'concave')
    n_orthogonal = sum(1 for rel in adj_segments if rel.get('relation') == 'orthogonal')
    
    features[14] = float(n_convex) / 10.0
    features[15] = float(n_concave) / 10.0
    features[16] = float(n_orthogonal) / 10.0
    
    contains = topology.get('contains', [])
    inside = topology.get('inside', [])
    features[17] = 1.0 if len(contains) > 0 else 0.0
    features[18] = 1.0 if len(inside) > 0 else 0.0
    features[19] = confidence
    
    return features

def normalize_2d_features_on_the_fly(node_data: dict) -> list:
    """
    Нормализует 2D признаки: value, tolerance, position, visual_features.
    Возвращает список из 260 элементов (4 + 256).
    """
    def safe_log(val):
        try:
            return math.log1p(max(0.0, float(val)))
        except (ValueError, TypeError):
            return 0.0
    val = float(node_data.get('value', 0.0))
    tol = float(node_data.get('tolerance', {}).get('value', 0.0))
    pos = node_data.get('position_2d', [0.0, 0.0])
    visual_emb = node_data.get('visual_features', [0.0]*256)
    
    # Нормализация числовых признаков
    val_norm = safe_log(val) if val > 0 else 0.0
    tol_norm = safe_log(tol) if tol > 0 else 0.0
    pos_x = float(pos[0]) / 1000.0  # Предполагаем, что координаты в мм, нормализуем к ~[-1, 1]
    pos_y = float(pos[1]) / 1000.0
    
    # Нормализация visual features (L2 нормализация)
    visual_emb_float = [float(v) for v in visual_emb]
    visual_norm = math.sqrt(sum(v*v for v in visual_emb_float) + 1e-8)
    visual_emb_normalized = [v / visual_norm for v in visual_emb_float]
    
    return [val_norm, tol_norm, pos_x, pos_y] + visual_emb_normalized

def load_graph_to_hetero_data(json_path: str) -> Optional[HeteroData]:
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception:
        return None
        
    nodes_2d = data.get('nodes_2d', {})
    nodes_3d = data.get('nodes_3d', {})
    edges = data.get('edges', [])
    
    if not nodes_2d or not nodes_3d:
        return None
        
    n2d_ids = list(nodes_2d.keys())
    n3d_ids = list(nodes_3d.keys())
    id2d_to_idx = {nid: i for i, nid in enumerate(n2d_ids)}
    id3d_to_idx = {nid: i for i, nid in enumerate(n3d_ids)}
    
    x_2d_list = []
    for nid in n2d_ids:
        node = nodes_2d[nid]
        features_2d = normalize_2d_features_on_the_fly(node)
        x_2d_list.append(features_2d)
        
    x_3d_list = []
    for nid in n3d_ids:
        node = nodes_3d[nid]
        features = normalize_3d_features_on_the_fly(node)
        if len(features) != 24:
            features = [0.0] * 24
        x_3d_list.append(features)
        
    hetero_data = HeteroData()
    hetero_data['2d'].x = torch.tensor(x_2d_list, dtype=torch.float32)
    hetero_data['3d'].x = torch.tensor(x_3d_list, dtype=torch.float32)
    
    edge_dict = {}
    corr_count = 0
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
            weight = float(edge.get('weight', 0.0))
            tol_status = edge.get('features', {}).get('tolerance_status', 'OUT_OF_TOLERANCE')
            if weight >= 0.8 and tol_status == 'IN_TOLERANCE': # Смягчили порог для обучения
                key = ('2d', 'correspondence', '3d')
                if key not in edge_dict: edge_dict[key] = [[], []]
                if src in id2d_to_idx and dst in id3d_to_idx:
                    edge_dict[key][0].append(id2d_to_idx[src])
                    edge_dict[key][1].append(id3d_to_idx[dst])
                corr_count += 1
                
    if corr_count == 0:
        return None
        
    for (src_type, rel_type, dst_type), indices in edge_dict.items():
        if len(indices[0]) > 0:
            hetero_data[src_type, rel_type, dst_type].edge_index = torch.tensor(indices, dtype=torch.long)
            
    return hetero_data

class HybridGraphDataset(Dataset):
    def __init__(self, data_dir: str, mode: str = 'train', val_split: float = 0.2):
        self.logger = logging.getLogger(__name__)
        self.mode = mode
        
        all_dirs = [d for d in Path(data_dir).iterdir() if d.is_dir()]
        self.logger.info(f"Найдено {len(all_dirs)} папок с данными. Сканирование...")
        
        valid_graphs = []
        in_tol_count = 0
        out_tol_count = 0
        
        for d in tqdm(all_dirs, desc="Загрузка графов"):
            for fname in ["graph_in_tol.json", "graph_out_tol.json"]:
                graph_path = d / "graphs" / fname
                if graph_path.exists():
                    data = load_graph_to_hetero_data(str(graph_path))
                    if data is not None:
                        valid_graphs.append(data)
                        if 'in_tol' in fname: in_tol_count += 1
                        else: out_tol_count += 1
                        
        self.logger.info(f"  Загружено всего: {len(valid_graphs)} графов (in_tol: {in_tol_count}, out_tol: {out_tol_count})")
        
        split_idx = int(len(valid_graphs) * (1 - val_split))
        if mode == 'train':
            self.data = valid_graphs[:split_idx]
        else:
            self.data = valid_graphs[split_idx:]
            
        self.logger.info(f"Режим {mode}: {len(self.data)} графов.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

# ==============================================================================
# 4. ЦИКЛ ОБУЧЕНИЯ И ОЦЕНКИ
# ==============================================================================
def train_epoch(model, loader, criterion, optimizer, device, writer, epoch):
    model.train()
    total_loss = 0
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, batch in enumerate(pbar):
        batch = batch.to(device)
        optimizer.zero_grad()
        
        z_2d, z_3d = model(batch.x_dict, batch.edge_index_dict)
        
        corr_key = ('2d', 'correspondence', '3d')
        if corr_key in batch.edge_index_dict and batch.edge_index_dict[corr_key].shape[1] > 0:
            corr_edges = batch.edge_index_dict[corr_key]
            loss = criterion(z_2d, z_3d, corr_edges)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            
            if batch_idx % 10 == 0:
                step = epoch * len(loader) + batch_idx
                writer.add_scalar('Loss/train_step', loss.item(), step)
        else:
            pbar.set_postfix({'loss': "0.0000 (no corr)"})
            
    return total_loss / max(1, len(loader))

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    valid_graphs_count = 0
    
    for batch in tqdm(loader, desc="Validation", leave=False):
        batch = batch.to(device)
        z_2d, z_3d = model(batch.x_dict, batch.edge_index_dict)
        
        corr_key = ('2d', 'correspondence', '3d')
        if corr_key in batch.edge_index_dict and batch.edge_index_dict[corr_key].shape[1] > 0:
            corr_edges = batch.edge_index_dict[corr_key]
            loss = criterion(z_2d, z_3d, corr_edges)
            total_loss += loss.item()
            valid_graphs_count += 1
            
    return total_loss / max(1, valid_graphs_count)

@torch.no_grad()
def evaluate_recall_at_1(model, loader, device):
    """
    Вычисляет Recall@1: долю случаев, когда модель поставила правильный 3D-узел 
    на 1-е место по косинусному сходству для данного 2D-узла.
    """
    model.eval()
    total_matches = 0
    total_edges = 0
    
    for batch in tqdm(loader, desc="Evaluating Recall@1", leave=False):
        batch = batch.to(device)
        
        z_2d, z_3d = model(batch.x_dict, batch.edge_index_dict)
        
        corr_key = ('2d', 'correspondence', '3d')
        if corr_key in batch.edge_index_dict:
            gt_edges = batch.edge_index_dict[corr_key]
            
            # Матрица сходства и предсказания
            sim = torch.matmul(z_2d, z_3d.T)
            pred_3d_indices = torch.argmax(sim, dim=1)
            
            # Считаем совпадения
            matches = (pred_3d_indices[gt_edges[0]] == gt_edges[1]).sum().item()
            total_matches += matches
            total_edges += gt_edges.shape[1]
            
    return total_matches / max(1, total_edges)

# ==============================================================================
# 5. MAIN
# ==============================================================================
def main():
    parser = argparse.ArgumentParser(description="Training HybridGNN")
    parser.add_argument('--data_dir', type=str, default="data/processed_full")
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=64)
    args = parser.parse_args()
    
    config = TrainConfig(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dim=args.hidden_dim
    )
    
    logger = setup_logging(config.log_dir)
    logger.info(f"Используется устройство: {config.device}")
    
    writer = SummaryWriter(log_dir=config.log_dir)
    
    train_dataset = HybridGraphDataset(config.data_dir, mode='train', val_split=config.val_split)
    val_dataset = HybridGraphDataset(config.data_dir, mode='val', val_split=config.val_split)
    
    train_loader = PyGDataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader = PyGDataLoader(val_dataset, batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    sample_2d_dim = train_dataset[0]['2d'].x.shape[1]
    sample_3d_dim = train_dataset[0]['3d'].x.shape[1]
    logger.info(f"Размерность 2D признаков: {sample_2d_dim}, 3D признаков: {sample_3d_dim}")
    
    model = HybridGNN(
        input_dim_2d=sample_2d_dim,
        input_dim_3d=sample_3d_dim,
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers
    ).to(config.device)
    
    criterion = NTXentLoss(temperature=config.temperature).to(config.device)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
    
    best_recall_at_1 = 0.0
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    
    logger.info("Начало обучения...")
    for epoch in range(1, config.epochs + 1):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, config.device, writer, epoch)
        val_loss = validate(model, val_loader, criterion, config.device)
        
        # === ВЫЧИСЛЕНИЕ Recall@1 ===
        recall_at_1 = evaluate_recall_at_1(model, val_loader, config.device)
        
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        logger.info(
            f"Epoch {epoch:03d} | "
            f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"Recall@1: {recall_at_1:.4f} | LR: {current_lr:.2e}"
        )
        
        writer.add_scalar('Loss/train_epoch', train_loss, epoch)
        writer.add_scalar('Loss/val_epoch', val_loss, epoch)
        writer.add_scalar('Metrics/Recall@1', recall_at_1, epoch)
        writer.add_scalar('LearningRate', current_lr, epoch)
        
        # Сохраняем лучшую модель по Recall@1
        if recall_at_1 > best_recall_at_1:
            best_recall_at_1 = recall_at_1
            checkpoint_path = Path(config.checkpoint_dir) / "best_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'recall_at_1': best_recall_at_1,
                'config': vars(config)
            }, checkpoint_path)
            logger.info(f" Сохранена лучшая модель (Recall@1: {best_recall_at_1:.4f})")

    writer.close()
    logger.info("Обучение завершено!")

if __name__ == '__main__':
    main()