#!/usr/bin/env python3
"""
Robust Hybrid Graph Construction.
1. Удалена ложная 2D-топология по расстоянию.
2. Умный поиск кандидатов (_get_3d_candidates) вместо первого попавшегося ключа.
3. Spatial weight снижен до 0.05.
4. Жадный алгоритм заменен на Hungarian algorithm (scipy.optimize).
5. Сохранение relation (concave/convex/orthogonal) как отдельного типа ребра для RGCN.
"""
import json
import numpy as np
from typing import List, Dict, Tuple
from pathlib import Path
from scipy.optimize import linear_sum_assignment # Проблема №4: Венгерский алгоритм

class TopologyAwareMatcher:
    def __init__(self, weights: Dict[str, float] = None):
        self.weights = weights or {
            'type': 0.20,
            'geometry': 0.75,  # Увеличен, так как это самый надежный признак
            'spatial': 0.01,   # ПРОБЛЕМА №5: Снижен до минимума до реализации мультивидовой проекции
            'topology': 0.04
        }
        self.all_2d_nodes = []

    def _get_3d_candidates(self, node_3d: Dict, hint_2d_text: str) -> List[float]:
        """Проблема №2: Возвращает ВСЕ возможные числовые значения для сравнения."""
        mv = node_3d.get('measured_value', {})
        if not isinstance(mv, dict):
            return [float(mv)] if isinstance(mv, (int, float)) else [0.0]
            
        vals = []
        for k, v in mv.items():
            if not isinstance(v, (int, float)):
                continue
            # Если это радиус, а на чертеже ищем диаметр (или наоборот)
            if 'radius' in k.lower() and ('⌀' in hint_2d_text or 'diameter' in hint_2d_text.lower()):
                vals.append(float(v) * 2.0)
            elif 'diameter' in k.lower() and 'r' in hint_2d_text.lower() and '⌀' not in hint_2d_text:
                vals.append(float(v) / 2.0)
            else:
                vals.append(float(v))
        return vals if vals else [0.0]

    def _get_2d_value_and_tol(self, node_2d: Dict) -> Tuple[float, float]:
        val = float(node_2d.get('value', 0.0))
        tol = node_2d.get('tolerance', {})
        tol_val = float(tol.get('value', 0.0))
        if tol_val == 0.0 and val > 0:
            tol_val = val * 0.05
        return val, tol_val

    def _calc_geometry_score(self, node_2d: Dict, node_3d: Dict) -> float:
        val_2d, tol = self._get_2d_value_and_tol(node_2d)
        hint_text = node_2d.get('text_content', '') + node_2d.get('semantic_info', {}).get('raw_text', '')
        candidates_3d = self._get_3d_candidates(node_3d, hint_text)
        
        if val_2d == 0:
            return 0.5
            
        # Проблема №2: Берем минимальную ошибку среди всех кандидатов
        min_error = min(abs(val_2d - c) for c in candidates_3d)
        
        if min_error <= tol:
            return 1.0
        elif min_error <= tol * 2.0:
            return 0.7
        else:
            return max(0.0, 0.7 - (min_error - tol * 2.0) / val_2d)

    def _calc_topology_score(self, node_2d: Dict, node_3d: Dict, edges_2d: List[Dict], nodes_3d_map: Dict) -> float:
        # Поскольку 2D-топология удалена, этот скор теперь опирается только на наличие 3D-соседей
        # как мягкий регуляризатор, но не как жесткое правило.
        n3d_ids_raw = node_3d.get('topology', {}).get('adjacent_segments', [])
        n3d_ids = [item.get('segment') if isinstance(item, dict) else item for item in n3d_ids_raw if item]
        
        if not n3d_ids:
            return 0.5
            
        # Если у 3D-узла есть богатая топология, это немного повышает его надежность
        return min(1.0, 0.5 + 0.1 * len(n3d_ids))

    def _get_semantic_mapping(self, node_2d: Dict) -> List[str]:
        cat = node_2d.get('semantic_info', {}).get('feature_category', '').lower()
        text = node_2d.get('text_content', '').lower()
        
        if 'hole' in cat or '⌀' in text:
            return ['through_hole', 'blind_hole', 'cylinder', 'sphere']
        elif 'datum' in cat:
            return ['plane', 'cylinder']
        elif 'width' in cat or 'slot' in cat:
            return ['slot', 'pocket']
        elif 'chamfer' in cat or 'r' in text:
            return ['chamfer', 'fillet', 'sphere', 'cone']
        return ['plane', 'pocket', 'cylinder', 'cone', 'through_hole', 'blind_hole']

    def match(self, nodes_2d: List[Dict], nodes_3d: List[Dict], edges_2d: List[Dict]) -> List[Dict]:
        self.all_2d_nodes = nodes_2d
        nodes_3d_map = {n['id']: n for n in nodes_3d}
        
        n2d_count = len(nodes_2d)
        n3d_count = len(nodes_3d)
        
        # Защита от пустых списков
        if n2d_count == 0 or n3d_count == 0:
            return []
            
        cost_matrix = np.full((n2d_count, n3d_count), np.inf)
        score_matrix = np.zeros((n2d_count, n3d_count))
        
        for i, n2d in enumerate(nodes_2d):
            allowed_types = self._get_semantic_mapping(n2d)
            hint_text = n2d.get('text_content', '') + n2d.get('semantic_info', {}).get('raw_text', '')
            
            for j, n3d in enumerate(nodes_3d):
                if n3d.get('type') not in allowed_types:
                    continue
                
                s_type = 1.0
                s_geom = self._calc_geometry_score(n2d, n3d)
                
                pos_2d = np.array(n2d.get('position_2d', [0.0, 0.0]))
                pos_3d = np.array(n3d.get('position_3d', [0.0, 0.0, 0.0])[:2])
                dist = np.linalg.norm(pos_2d - pos_3d) / 100.0
                s_spatial = max(0.0, 1.0 - dist)
                
                s_topo = self._calc_topology_score(n2d, n3d, edges_2d, nodes_3d_map)
                
                total_score = (
                    self.weights['type'] * s_type +
                    self.weights['geometry'] * s_geom +
                    self.weights['spatial'] * s_spatial +
                    self.weights['topology'] * s_topo
                )
                
                score_matrix[i, j] = total_score
                # Если скор выше порога, записываем стоимость (алгоритм минимизирует cost)
                if total_score >= 0.6:
                    cost_matrix[i, j] = 1.0 - total_score

        if np.isinf(cost_matrix).all():
            return []
            
        # Проблема №4: Глобально оптимальное бипартитное сопоставление
        try:
            row_ind, col_ind = linear_sum_assignment(cost_matrix)
        except Exception as e:
            print(f"  [Warning] linear_sum_assignment failed ({e}). Falling back to greedy matching.")
            # Fallback: жадное сопоставление по убыванию скора
            indices = np.argsort(score_matrix.flatten())[::-1]
            rows, cols = np.unravel_index(indices, score_matrix.shape)
            row_ind, col_ind = [], []
            used_rows, used_cols = set(), set()
            for r, c in zip(rows, cols):
                if score_matrix[r, c] >= 0.6 and r not in used_rows and c not in used_cols:
                    row_ind.append(r)
                    col_ind.append(c)
                    used_rows.add(r)
                    used_cols.add(c)
            row_ind = np.array(row_ind)
            col_ind = np.array(col_ind)
        
        final_edges = []
        for i, j in zip(row_ind, col_ind):
            if cost_matrix[i, j] == np.inf:
                continue  # Пропускаем пары, которые алгоритм был вынужден выбрать, но они ниже порога
                
            n2d = nodes_2d[i]
            n3d = nodes_3d[j]
            score = score_matrix[i, j]
            
            val_2d, tol = self._get_2d_value_and_tol(n2d)
            candidates = self._get_3d_candidates(n3d, n2d.get('text_content', '') + n2d.get('semantic_info', {}).get('raw_text', ''))
            
            # Находим значение из кандидатов, которое дало минимальную ошибку
            best_val_3d = min(candidates, key=lambda x: abs(val_2d - x))
            
            deviation = abs(val_2d - best_val_3d)
            status = 'IN_TOLERANCE' if deviation <= tol else 'OUT_OF_TOLERANCE'
            
            final_edges.append({
                'src': n2d['id'],
                'dst': n3d['id'],
                'type': 'correspondence',
                'weight': round(float(score), 3),
                'features': {
                    'deviation': round(float(deviation), 3),
                    'tolerance_status': status,
                    'matching_scores': {
                        'geom': round(float(self._calc_geometry_score(n2d, n3d)), 3),
                        'spatial': round(float(max(0.0, 1.0 - np.linalg.norm(np.array(n2d.get('position_2d', [0.0, 0.0])) - np.array(n3d.get('position_3d', [0.0, 0.0, 0.0])[:2])) / 100.0)), 3)
                    }
                }
            })
            
        return final_edges


class HybridGraph:
    def __init__(self):
        self.nodes_2d = {}
        self.nodes_3d = {}
        self.edges = []

    def add_3d_adjacency_edges(self, features_3d: List[Dict]):
        """Проблема №7: Добавляем только ОДНО направление. PyG ToUndirected() сам создаст обратное."""
        for feat in features_3d:
            src_id = feat['id']
            raw_neighbors = feat.get('topology', {}).get('adjacent_segments', [])
            
            for item in raw_neighbors:
                if isinstance(item, dict):
                    dst_id = item.get('segment')
                    relation = item.get('relation', 'unknown')
                else:
                    dst_id = f"seg_{int(item):03d}"
                    relation = 'unknown'
                    
                if not dst_id:
                    continue
                
                # Добавляем только src -> dst. Не добавляем обратное ребро вручную!
                edge_type = f"adjacent_3d_{relation}"
                
                self.edges.append({
                    'src': src_id,
                    'dst': dst_id,
                    'type': edge_type,
                    'weight': 1.0,
                    'features': {'relation': relation}
                })
                
    def add_2d_adjacency_edges(self, features_2d: List[Dict]):
        """Проблема №1: Полностью отключено. Лучше нет топологии, чем случайный мусор."""
        pass

    def save(self, filepath: str):
        # Подсчет статистики по новым типам ребер
        edge_counts = {}
        for e in self.edges:
            t = e['type']
            edge_counts[t] = edge_counts.get(t, 0) + 1
            
        # Для неориентированных ребер делим на 2
        pretty_counts = {}
        for k, v in edge_counts.items():
            if k.startswith('adjacent_3d_'):
                pretty_counts[k] = v // 2
            else:
                pretty_counts[k] = v

        data = {
            'nodes_2d': self.nodes_2d,
            'nodes_3d': self.nodes_3d,
            'edges': self.edges,
            'metadata': {
                'num_2d_nodes': len(self.nodes_2d),
                'num_3d_nodes': len(self.nodes_3d),
                'num_edges': len(self.edges),
                'edge_type_counts': pretty_counts
            }
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
            
        print(f"  Hybrid graph saved to {filepath}")
        print(f"  Nodes: {len(self.nodes_2d)} (2D), {len(self.nodes_3d)} (3D)")
        print(f"  Edge Types: {pretty_counts}")


def construct_graph(f2d_path: str, f3d_path: str, output_path: str) -> HybridGraph:
    with open(f2d_path, 'r', encoding='utf-8') as f:
        data_2d = json.load(f)
    with open(f3d_path, 'r', encoding='utf-8') as f:
        data_3d = json.load(f)
        
    features_2d = data_2d.get('features', [])
    features_3d = data_3d.get('features', [])
    
    graph = HybridGraph()
    graph.nodes_2d = {f['id']: f for f in features_2d}
    graph.nodes_3d = {f['id']: f for f in features_3d}
    
    print("  [Graph] Building 3D adjacency edges (preserving convex/concave/orthogonal)...")
    graph.add_3d_adjacency_edges(features_3d)
    
    print("  [Graph] 2D adjacency disabled (distance != semantic relation)...")
    graph.add_2d_adjacency_edges(features_2d)
    
    print("  [Graph] Running Hungarian algorithm for optimal correspondence matching...")
    matcher = TopologyAwareMatcher()
    corr_edges = matcher.match(features_2d, features_3d, [])
    graph.edges.extend(corr_edges)
    
    graph.save(output_path)
    return graph