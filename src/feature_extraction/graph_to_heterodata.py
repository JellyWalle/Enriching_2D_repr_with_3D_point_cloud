import json
import torch
import numpy as np
from pathlib import Path
from torch_geometric.data import HeteroData

def load_hybrid_graph_to_heterodata(json_path: str) -> HeteroData:
    """
    Этап 4.3: Кодирование признаков узлов и рёбер.
    Преобразует hybrid_graph.json в PyG HeteroData.
    """
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    nodes_2d = data.get('nodes_2d', {})
    nodes_3d = data.get('nodes_3d', {})
    edges = data.get('edges', [])

    # Маппинг ID в индексы
    n2d_ids = list(nodes_2d.keys())
    n3d_ids = list(nodes_3d.keys())
    id2d_to_idx = {nid: i for i, nid in enumerate(n2d_ids)}
    id3d_to_idx = {nid: i for i, nid in enumerate(n3d_ids)}

    # ==========================================
    # 1. Формирование признаков узлов (Node Features)
    # ==========================================
    x_2d_list = []
    for nid in n2d_ids:
        node = nodes_2d[nid]
        # Базовые признаки (4D): value, tol_value, x, y
        val = float(node.get('value', 0.0))
        tol = float(node.get('tolerance', {}).get('value', 0.0))
        pos = node.get('position_2d', [0.0, 0.0])
        x, y = pos[0], pos[1]
        
        # Эмбеддинги (768D + 256D = 1024D)
        text_emb = node.get('text_embedding', [])
        if len(text_emb) != 768:
            text_emb = [0.0] * 768
            
        visual_emb = node.get('visual_features', [])
        if len(visual_emb) != 256:
            visual_emb = [0.0] * 256
            
        x_2d_list.append([val, tol, x, y] + text_emb + visual_emb)

    x_3d_list = []
    for nid in n3d_ids:
        node = nodes_3d[nid]
        # Используем предварительно вычисленный 18-мерный вектор из features3d.py
        # Если его нет (старый формат), используем заглушку, но план требует 18D
        features = node.get('node_features', [0.0] * 18)
        if len(features) != 18:
            features = [0.0] * 18 # Fallback
        x_3d_list.append(features)

    # ==========================================
    # 2. Формирование рёбер (Edge Indices)
    # ==========================================
    edge_dict = {} # Ключ: (src_type, rel_type, dst_type), Значение: [[src], [dst]]

    for edge in edges:
        src, dst, rel = edge['src'], edge['dst'], edge['type']
        
        if rel == 'adjacent_2d':
            key = ('2d', 'adjacent_2d', '2d')
            if key not in edge_dict: edge_dict[key] = [[], []]
            if src in id2d_to_idx and dst in id2d_to_idx:
                edge_dict[key][0].append(id2d_to_idx[src])
                edge_dict[key][1].append(id2d_to_idx[dst])
                
        elif rel.startswith('adjacent_3d_'): # convex, concave, orthogonal, unknown
            key = ('3d', rel, '3d')
            if key not in edge_dict: edge_dict[key] = [[], []]
            if src in id3d_to_idx and dst in id3d_to_idx:
                edge_dict[key][0].append(id3d_to_idx[src])
                edge_dict[key][1].append(id3d_to_idx[dst])
                
        elif rel == 'correspondence':
            key = ('2d', 'correspondence', '3d')
            if key not in edge_dict: edge_dict[key] = [[], []]
            if src in id2d_to_idx and dst in id3d_to_idx:
                edge_dict[key][0].append(id2d_to_idx[src])
                edge_dict[key][1].append(id3d_to_idx[dst])

    # ==========================================
    # 3. Сборка HeteroData объекта
    # ==========================================
    hetero_data = HeteroData()
    hetero_data['2d'].x = torch.tensor(x_2d_list, dtype=torch.float32)
    hetero_data['3d'].x = torch.tensor(x_3d_list, dtype=torch.float32)

    for (src_type, rel_type, dst_type), indices in edge_dict.items():
        if len(indices[0]) > 0:
            edge_index = torch.tensor(indices, dtype=torch.long)
            hetero_data[src_type, rel_type, dst_type].edge_index = edge_index
            
            # Добавляем атрибуты рёбер для correspondence (Этап 4.3)
            if rel_type == 'correspondence':
                # Собираем deviation и weight для этих рёбер
                devs, weights = [], []
                # Создаем быстрый доступ к рёбрам
                edge_map = {(e['src'], e['dst']): e for e in edges if e['type'] == 'correspondence'}
                for i in range(edge_index.shape[1]):
                    src_id = n2d_ids[edge_index[0, i].item()]
                    dst_id = n3d_ids[edge_index[1, i].item()]
                    e_data = edge_map.get((src_id, dst_id), {})
                    devs.append(float(e_data.get('features', {}).get('deviation', 0.0)))
                    weights.append(float(e_data.get('weight', 1.0)))
                
                hetero_data['2d', 'correspondence', '3d'].deviation = torch.tensor(devs, dtype=torch.float32).unsqueeze(1)
                hetero_data['2d', 'correspondence', '3d'].weight = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)

    return hetero_data