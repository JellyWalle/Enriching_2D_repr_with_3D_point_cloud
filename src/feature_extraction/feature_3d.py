#!/usr/bin/env python3
"""
3D Feature Extraction from Point Clouds.
Исправлено: сегментация, сериализация JSON, конфигурация, топология.
Зависимости: numpy, scipy, open3d, scikit-learn, json
"""
from collections import deque
import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import open3d as o3d
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

# ==================== Утилиты ====================
class NumpyEncoder(json.JSONEncoder):
    """Автоматически преобразует numpy-типы в стандартные Python-типы для JSON."""
    def default(self, obj):
        if isinstance(obj, np.integer): return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        return super().default(obj)

class Node3D:
    def __init__(self, node_id: str, feature_type: str, measured_value: float,
                 position_3d: Tuple[float, float, float], normals: List[float] = None,
                 curvature: float = 0.0, topology: Dict = None, geometry_features: Dict = None,
                 spatial_encoding: Dict = None):
        self.id = node_id
        self.type = 'geometry'
        self.feature_type = feature_type
        self.measured_value = measured_value
        self.position_3d = position_3d
        self.normals = normals or [0.0, 0.0, 1.0]
        self.curvature = curvature
        self.topology = topology or {
            'adjacent_faces': [], 'parent_feature': None, 'child_features': [],
            'connected_edges': [], 'surface_connectivity': []
        }
        self.geometry_features = geometry_features or {
            'surface_area': 0.0, 'volume': 0.0, 'bounding_box': None,
            'principal_directions': None, 'aspect_ratio': 1.0,
            'circularity': 1.0, 'planarity': 0.0
        }
        self.spatial_encoding = spatial_encoding or {
            'relative_position': None, 'distance_to_origin': 0.0,
            'orientation_angles': None, 'layer_index': 0, 'quadrant': None
        }
        self.point_indices = []
        self.num_points = 0
        self.confidence = 1.0
        self.unit = 'mm'
        self.unit_conversion = {}
        self.corresponding_2d_nodes = []

    def to_dict(self) -> Dict:
        # Явное преобразование для гарантии совместимости
        return {
            'id': self.id, 'type': self.type, 'feature_type': self.feature_type,
            'measured_value': float(self.measured_value), 'unit': self.unit,
            'unit_conversion': self.unit_conversion,
            'position_3d': [float(x) for x in self.position_3d],
            'normals': [float(x) for x in self.normals], 'curvature': float(self.curvature),
            'topology': self.topology, 'geometry_features': self.geometry_features,
            'spatial_encoding': self.spatial_encoding,
            'point_indices': [int(x) for x in self.point_indices], 
            'num_points': int(self.num_points), 'confidence': float(self.confidence),
            'corresponding_2d_nodes': self.corresponding_2d_nodes
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Node3D':
        node = cls(
            node_id=data['id'], feature_type=data.get('feature_type', 'unknown'),
            measured_value=data.get('measured_value', 0.0),
            position_3d=tuple(data.get('position_3d', [0,0,0])),
            normals=data.get('normals'), curvature=data.get('curvature', 0.0),
            topology=data.get('topology'), geometry_features=data.get('geometry_features'),
            spatial_encoding=data.get('spatial_encoding')
        )
        node.point_indices = data.get('point_indices', [])
        node.num_points = data.get('num_points', 0)
        node.confidence = data.get('confidence', 1.0)
        node.unit = data.get('unit', 'mm')
        node.unit_conversion = data.get('unit_conversion', {})
        node.corresponding_2d_nodes = data.get('corresponding_2d_nodes', [])
        return node

class PointCloudPreprocessor:
    def __init__(self, config: Dict = None):
        defaults = {'target_points': 4096, 'normalize_center': True, 'normalize_scale': 1.0, 'outlier_threshold': 2.0, 'outlier_k': 20}
        defaults.update(config or {})
        self.cfg = defaults

    def preprocess(self, points: np.ndarray) -> np.ndarray:
        if len(points) == 0: return points
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        k = self.cfg.get('outlier_k_neighbors', self.cfg.get('outlier_k', 20))
        std = self.cfg.get('outlier_threshold', 2.0)
        if len(points) > k:
            pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=k, std_ratio=std)
            
        points = np.asarray(pcd.points)
        if len(points) == 0: return np.empty((0, 3))
        
        if self.cfg.get('normalize_center', True):
            points -= points.mean(axis=0)
        scale = self.cfg.get('normalize_scale', 1.0)
        if scale > 0:
            max_dist = np.linalg.norm(points, axis=1).max()
            if max_dist > 1e-6: points = points * scale / max_dist
            
        target = self.cfg.get('target_points', 4096)
        #if len(points) > target:
        #    points = points[np.random.choice(len(points), target, replace=False)]
        return points

import os
import numpy as np
import open3d as o3d

from scipy.spatial import KDTree


def _normalize(v):
    vmin = np.min(v)
    vmax = np.max(v)
    return (v - vmin) / (vmax - vmin + 1e-9)


class GeometricSegmenter:

    def __init__(self, config=None):
        defaults = {
            'normal_tolerance_deg': 15.0,
            'seed_normal_tolerance_deg': 75.0,
            'curvature_tolerance': None,
            'min_segment_size': 30,
            'k_neighbors': 10,
            'debug': True,
            'debug_dir': './debug_segmentation'
        }
        defaults.update(config or {})
        self.cfg = defaults

    def segment(self, points: np.ndarray):

        os.makedirs(self.cfg['debug_dir'], exist_ok=True)

        min_size = self.cfg['min_segment_size']

        if len(points) < min_size:
            return {
                'seg_0': {
                    'feature_type': 'unknown',
                    'nodes': list(range(len(points))),
                    'confidence': 0.3
                }
            }

        tree = KDTree(points)

        k = min(self.cfg['k_neighbors'] + 1, len(points))
        dists, _ = tree.query(points, k=k)

        avg_spacing = float(np.mean(dists[:, -1]))

        print(f'\n[DEBUG] avg_spacing = {avg_spacing:.6f}')

        normal_radius = avg_spacing * 2.0

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        pcd.estimate_normals(
            search_param=o3d.geometry.KDTreeSearchParamHybrid(
                radius=normal_radius,
                max_nn=15
            )
        )
        pcd.orient_normals_consistent_tangent_plane(30)

        normals = np.asarray(pcd.normals)

        curvatures = self._compute_curvature(points, normals, avg_spacing)

        # =====================
        # ADAPTIVE THRESHOLDS
        # =====================

        if self.cfg['curvature_tolerance'] is None:
            curv_tol = np.percentile(curvatures, 75) * 0.35
        else:
            curv_tol = self.cfg['curvature_tolerance']

        print(f'[DEBUG] curvature_tol = {curv_tol:.6f}')

        local_cos = np.cos(np.radians(self.cfg['normal_tolerance_deg']))
        global_cos = np.cos(np.radians(self.cfg['seed_normal_tolerance_deg']))

        # =====================
        # DEBUG STATISTICS
        # =====================

        print('\n[DEBUG] CURVATURE PERCENTILES')
        print(np.percentile(curvatures, [1,5,10,25,50,75,90,95,99]))

        angles = []

        for i in range(len(points)):
            _, idx = tree.query(points[i], k=min(10, len(points)))

            for j in idx[1:]:
                dot = np.clip(np.dot(normals[i], normals[j]), -1, 1)
                angle = np.degrees(np.arccos(abs(dot)))
                angles.append(angle)

        print('\n[DEBUG] NORMAL ANGLE PERCENTILES')
        print(np.percentile(angles, [50,75,90,95,99]))

        # =====================
        # REGION GROWING
        # =====================

        sorted_indices = np.argsort(curvatures)

        visited = set()
        segments = {}
        seg_id = 0

        from collections import deque

        for seed in sorted_indices:

            seed = int(seed)

            if seed in visited:
                continue

            queue = deque([seed])

            cluster = []

            while queue:

                curr = queue.popleft()

                if curr in visited:
                    continue

                visited.add(curr)

                cluster.append(curr)

                _, nb_indices = tree.query(
                    points[curr],
                    k=min(self.cfg['k_neighbors'] + 1, len(points))
                )

                for nb in map(int, nb_indices[1:]):

                    if nb in visited:
                        continue

                    # spatial continuity
                    dist = np.linalg.norm(points[curr] - points[nb])

                    if dist > avg_spacing * 4.0:
                        continue

                    # normal similarity
                    dot_local = abs(np.dot(normals[curr], normals[nb]))

                    if dot_local < local_cos:
                        continue

                    # curvature similarity
                    curv_diff = abs(
                        float(curvatures[curr]) -
                        float(curvatures[nb])
                    )

                    if curv_diff > curv_tol:
                        continue

                    queue.append(nb)

            # сохраняем сегмент
            if len(cluster) >= min_size:

                arr = np.array(cluster)

                feature_type = self._classify_segment(
                    points[arr],
                    curvatures[arr],
                    normals[arr]
                )

                segments[f'seg_{seg_id}'] = {
                    'feature_type': feature_type,
                    'nodes': arr.tolist(),
                    'confidence': 0.85,
                    'curvature_mean': float(curvatures[arr].mean())
                }

                print(
                    f'[SEGMENT] seg_{seg_id} | '
                    f'size={len(arr)} | '
                    f'type={feature_type} | '
                    f'curv={curvatures[arr].mean():.5f}'
                )

                seg_id += 1


        # =====================
        # DEBUG SEGMENTS
        # =====================

        if self.cfg['debug'] and len(segments) > 0:

            rng = np.random.default_rng(42)

            colors = np.zeros((len(points), 3))

            for sid, seg in segments.items():
                color = rng.random(3)
                colors[np.array(seg['nodes'])] = color

            dbg = o3d.geometry.PointCloud()
            dbg.points = o3d.utility.Vector3dVector(points)
            dbg.colors = o3d.utility.Vector3dVector(colors)

            o3d.io.write_point_cloud(
                os.path.join(self.cfg['debug_dir'], 'debug_segments.ply'),
                dbg
            )

        print(f'\n[DEBUG] TOTAL SEGMENTS = {len(segments)}')

        if not segments:
            return {
                'seg_0': {
                    'feature_type': 'unknown',
                    'nodes': list(range(len(points))),
                    'confidence': 0.3
                }
            }
        debugger = SegmentDebugger()
        debugger.dump_segments(
            points,
            normals,
            curvatures,
            segments,
            classifier_fn=self._classify_segment
            )
               

        return segments



    def _compute_curvature(self, points, normals, avg_spacing):

        tree = KDTree(points)

        radius_small = avg_spacing * 3.0
        radius_large = avg_spacing * 6.0

        curvatures = np.zeros(len(points))

        for i, p in enumerate(points):

            idx_small = tree.query_ball_point(p, radius_small)
            idx_large = tree.query_ball_point(p, radius_large)

            if len(idx_small) < 5:
                continue

            pts_small = points[idx_small]
            pts_large = points[idx_large]

            cov_small = np.cov((pts_small - pts_small.mean(axis=0)).T)
            cov_large = np.cov((pts_large - pts_large.mean(axis=0)).T)

            eig_small = np.sort(np.linalg.eigvalsh(cov_small))
            eig_large = np.sort(np.linalg.eigvalsh(cov_large))

            curv_small = eig_small[0] / (eig_small.sum() + 1e-9)
            curv_large = eig_large[0] / (eig_large.sum() + 1e-9)

            # multi-scale curvature
            curvatures[i] = 0.7 * curv_small + 0.3 * curv_large

        return curvatures

        
    def _classify_segment(self, pts, curvs, norms):

        if len(pts) < 5:
            return 'unknown'

        centroid = pts.mean(axis=0)
        centered = pts - centroid

        cov = np.cov(centered.T)
        eigvals = np.sort(np.linalg.eigvalsh(cov))[::-1]

        total = eigvals.sum() + 1e-9

        e0 = eigvals[0] / total
        e1 = eigvals[1] / total
        e2 = eigvals[2] / total

        mc = float(curvs.mean())

        normal_cov = np.cov(norms.T)
        normal_eigs = np.linalg.eigvalsh(normal_cov)
        
        normal_consistency = np.mean(
            np.abs(norms @ norms.mean(axis=0))
        )

        linearity = (e0 - e1) / (e0 + 1e-9)
        planarity = (e1 - e2) / (e0 + 1e-9)
        scattering = e2 / (e0 + 1e-9)

        print(
            f'[CLASSIFY] size={len(pts)} '
            f'e0={e0:.3f} '
            f'e1={e1:.3f} '
            f'e2={e2:.3f} '
            f'curv={mc:.5f} '
            f'norm={normal_consistency:.3f}'
        )

        # plane
        if planarity > 0.55 and mc < 0.03:
            return 'plane'

        # cylinder / hole
        if (
            linearity < 0.4 and
            planarity > 0.2 and
            scattering < 0.12 and
            normal_consistency > 0.85
        ):
            return 'hole'

        # edge / chamfer
        if (
            0.03 < mc < 0.08 and
            planarity > 0.3 and
            scattering < 0.15
        ):
            return 'chamfer'

        # pocket
        if scattering > 0.12:
            return 'pocket'

        return 'unknown'

class GeometricFeatureComputer:
    def __init__(self, config: Dict = None):
        defaults = {'normal_tol': 0.1, 'curvature_tol': 0.05, 'min_segment_size': 30, 'k': 15}
        defaults.update(config or {})
        self.cfg = defaults

    def compute_attributes(self, pts: np.ndarray, seg_info: Dict) -> Dict:
        if len(pts) == 0: return {}
        tree = KDTree(pts)
        normals = np.array([self._normal(p, tree) for p in pts])
        curvature = float(np.mean([self._curvature(p, tree, normals) for p in pts]))
        measured = self._measure(pts, seg_info['feature_type'])
        centroid = pts.mean(axis=0).tolist()
        geometry = self._geometry(pts)
        spatial = self._spatial(centroid)
        
        return {
            'normals': normals.mean(axis=0).tolist(), 'curvature': curvature,
            'measured_value': measured, 'centroid': centroid,
            'geometry': geometry, 'spatial': spatial
        }

    def _normal(self, p, tree):
        k = min(self.cfg['k'], len(tree.data))
        _, idx = tree.query(p, k=k)
        n = PCA(n_components=3).fit(tree.data[idx]).components_[2]
        return n if np.dot(n, p) >= 0 else -n

    def _curvature(self, p, tree, normals):
        _, idx = tree.query(p, k=min(self.cfg['k'], len(tree.data)))
        return float(np.mean(np.linalg.norm(normals[idx] - normals[idx].mean(axis=0), axis=1)))

    def _measure(self, pts, ftype):
        bb = pts.max(axis=0) - pts.min(axis=0)
        if ftype in ('hole', 'cylinder') and len(pts) >= 3:
            axis = PCA(n_components=3).fit(pts).components_[2]
            proj = pts - np.outer(pts @ axis, axis)
            return float(2 * np.linalg.norm(proj, axis=1).mean())
        return float(bb.max())

    def _geometry(self, pts):
        bb = pts.min(axis=0).tolist() + pts.max(axis=0).tolist()
        dims = pts.max(axis=0) - pts.min(axis=0)
        return {
            'bounding_box': bb, 'aspect_ratio': float(dims.max() / (dims[dims>0].min() + 1e-9)),
            'surface_area': float(len(pts) * 0.1), 'planarity': 0.0, 'circularity': 1.0
        }

    def _spatial(self, centroid):
        dist = float(np.linalg.norm(centroid))
        quad = 'Q1' if centroid[0]>=0 and centroid[1]>=0 else 'Q2' if centroid[0]<0 and centroid[1]>=0 else 'Q3' if centroid[0]<0 and centroid[1]<0 else 'Q4'
        return {'relative_position': centroid, 'distance_to_origin': dist, 'quadrant': quad, 'layer_index': int(centroid[2]//10)}



class SegmentDebugger:

    def __init__(self, out_dir="./debug_classifier"):
        self.out_dir = Path(out_dir)

        self.seg_dir = self.out_dir / "segments"
        self.meta_dir = self.out_dir / "metadata"

        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.meta_dir.mkdir(parents=True, exist_ok=True)

    def dump_segments(
        self,
        points,
        normals,
        curvatures,
        segments,
        classifier_fn=None
    ):

        summary = []

        for seg_id, seg in segments.items():

            idx = np.array(seg["nodes"])

            pts = points[idx]
            nrm = normals[idx]
            curv = curvatures[idx]

            metrics = self.compute_metrics(
                pts,
                nrm,
                curv
            )

            predicted = seg.get("feature_type", "unknown")

            if classifier_fn is not None:
                predicted = classifier_fn(
                    pts,
                    curv,
                    nrm
                )

            # ==========================
            # SAVE POINT CLOUD
            # ==========================

            pcd = o3d.geometry.PointCloud()

            pcd.points = o3d.utility.Vector3dVector(pts)

            colors = np.zeros((len(pts), 3))

            c = self.normalize(curv)

            colors[:, 0] = c
            colors[:, 2] = 1.0 - c

            pcd.colors = o3d.utility.Vector3dVector(colors)

            ply_path = self.seg_dir / f"{seg_id}.ply"

            o3d.io.write_point_cloud(
                str(ply_path),
                pcd
            )

            # ==========================
            # SAVE METADATA
            # ==========================

            metadata = {
                "segment_id": seg_id,
                "predicted_type": predicted,
                "manual_label": None,

                "num_points": int(len(pts)),

                "metrics": metrics,

                "curvature_stats": {
                    "min": float(np.min(curv)),
                    "max": float(np.max(curv)),
                    "mean": float(np.mean(curv)),
                    "std": float(np.std(curv)),
                    "p25": float(np.percentile(curv, 25)),
                    "p50": float(np.percentile(curv, 50)),
                    "p75": float(np.percentile(curv, 75)),
                }
            }

            json_path = self.meta_dir / f"{seg_id}.json"

            with open(json_path, "w") as f:
                json.dump(metadata, f, indent=2)

            summary.append({
                "segment_id": seg_id,
                "predicted_type": predicted,
                "num_points": int(len(pts)),
                **metrics
            })

            print(
                f"[DEBUG SEGMENT] {seg_id} | "
                f"type={predicted} | "
                f"pts={len(pts)}"
            )

        # ==========================
        # SAVE GLOBAL CSV-LIKE JSON
        # ==========================

        with open(self.out_dir / "summary.json", "w") as f:
            json.dump(summary, f, indent=2)

        print("\nSaved debug dataset:")
        print(self.out_dir)

    def compute_metrics(self, pts, normals, curvs):

        centroid = pts.mean(axis=0)

        centered = pts - centroid

        cov = np.cov(centered.T)

        eigvals = np.sort(
            np.linalg.eigvalsh(cov)
        )[::-1]

        total = eigvals.sum() + 1e-9

        e0 = eigvals[0] / total
        e1 = eigvals[1] / total
        e2 = eigvals[2] / total

        linearity = (e0 - e1) / (e0 + 1e-9)
        planarity = (e1 - e2) / (e0 + 1e-9)
        scattering = e2 / (e0 + 1e-9)

        mean_normal = normals.mean(axis=0)
        mean_normal /= (
            np.linalg.norm(mean_normal) + 1e-9
        )

        normal_consistency = np.mean(
            np.abs(normals @ mean_normal)
        )

        bbox = pts.max(axis=0) - pts.min(axis=0)

        aspect_ratio = (
            bbox.max() /
            (bbox.min() + 1e-9)
        )

        return {

            "e0": float(e0),
            "e1": float(e1),
            "e2": float(e2),

            "linearity": float(linearity),
            "planarity": float(planarity),
            "scattering": float(scattering),

            "normal_consistency": float(normal_consistency),

            "curvature_mean": float(np.mean(curvs)),
            "curvature_std": float(np.std(curvs)),

            "aspect_ratio": float(aspect_ratio),

            "bbox_x": float(bbox[0]),
            "bbox_y": float(bbox[1]),
            "bbox_z": float(bbox[2]),
        }

    def normalize(self, x):

        x = np.asarray(x)

        return (
            x - x.min()
        ) / (
            x.max() - x.min() + 1e-9
        )
    
class FeatureExtractor3D:
    """Основной пайплайн извлечения 3D признаков."""
    def __init__(self, config: Dict = None):
        defaults = {'target_points': 4096, 'min_segment_size': 30,
                    'normalize_center': True, 'normalize_scale': 1.0,
                    'outlier_threshold': 2.0, 'outlier_k': 20}
        defaults.update(config or {})
        self.cfg = defaults
        
        self.preprocessor = PointCloudPreprocessor(self.cfg)
        # Передаем конфиг сегментатору (он сам сольет с дефолтами)
        self.segmenter = GeometricSegmenter({
            'min_segment_size': self.cfg['min_segment_size'],
            'normal_tolerance_deg': self.cfg.get('normal_tolerance_deg', 15),
            'seed_normal_tolerance_deg': self.cfg.get('seed_normal_tolerance_deg', 75),
            'curvature_tolerance': self.cfg.get('curvature_tolerance', None),
            'k_neighbors': self.cfg.get('k_neighbors', 10),
            'debug': self.cfg.get('debug', True),
            'debug_dir': self.cfg.get('debug_dir', './debug_segmentation')
        })
        self.computer = GeometricFeatureComputer()
        self.model_centroid = None

    def extract_features(self, pointcloud_path: str) -> List[Node3D]:
        points = self._load(pointcloud_path)
        points = self.preprocessor.preprocess(points)
        print(f'[DEBUG] num_points after preprocessing: {len(points)}')
        if len(points) < 10: return []
        
        self.model_centroid = points.mean(axis=0)
        segments = self.segmenter.segment(points)
 
        # Построение топологии смежности
        seg_centroids = {}
        for sid, s in segments.items():
            seg_centroids[sid] = points[s['nodes']].mean(axis=0)
            
        nodes = []
        for i, (sid, s) in enumerate(segments.items()):
            pts = points[s['nodes']]
            attrs = self.computer.compute_attributes(pts, s)
            if attrs['measured_value'] <= 0: continue
            
            # Топология (смежные сегменты в радиусе 2*средний_размер)
            adj = []
            for other_sid, oc in seg_centroids.items():
                if sid == other_sid: continue
                if np.linalg.norm(seg_centroids[sid] - oc) < 2.0: adj.append(other_sid)
                
            node = Node3D(
                node_id=f'node3d_{i:03d}', feature_type=s['feature_type'],
                measured_value=attrs['measured_value'], position_3d=tuple(attrs['centroid']),
                normals=attrs['normals'], curvature=attrs['curvature'],
                topology={'adjacent_faces': adj, 'parent_feature': None, 'child_features': []},
                geometry_features=attrs['geometry'], spatial_encoding=attrs['spatial']
            )
            node.point_indices = s['nodes']
            node.num_points = len(s['nodes'])
            node.confidence = s.get('confidence', 0.8)
            node.unit_conversion = {'original': 'normalized', 'converted': 'mm'}
            nodes.append(node)
        return nodes

    def _load(self, path: str) -> np.ndarray:
        if path.endswith('.npy'): return np.load(path)
        pcd = o3d.io.read_point_cloud(path)
        return np.asarray(pcd.points)


def save_features(features: List[Node3D], out_path: str):
    with open(out_path, 'w') as f:
        json.dump({'num_features': len(features), 'features': [n.to_dict() for n in features]}, f, indent=2)

def load_features(in_path: str) -> List[Node3D]:
    with open(in_path, 'r') as f: data = json.load(f)
    return [Node3D.from_dict(d) for d in data.get('features', [])]