#!/usr/bin/env python3
"""
3D Feature Segmentation from STEP files.

Архитектура: B-Rep Topology Rule-Based Recognition (подход Joshi & Chang).
Скрипт загружает STEP-файл, извлекает топологию граней, распознает 
технологические признаки (отверстия, карманы, фаски и т.д.) на основе 
правил смежности, генерирует облако точек с привязкой к граням и 
формирует конфигурацию признаков с геометрическими и топологическими 
дескрипторами для последующей обработки графовыми нейронными сетями (GNN).
"""

import json
from collections import defaultdict
from typing import Dict, List, Set, Tuple

import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from scipy.spatial import KDTree
from sklearn.decomposition import PCA

# ==================== PythonOCC Imports ====================
from OCC.Core.BRep import BRep_Tool
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import (
    GeomAbs_BSplineSurface,
    GeomAbs_Cone,
    GeomAbs_Cylinder,
    GeomAbs_Plane,
    GeomAbs_Sphere,
    GeomAbs_Torus,
)
from OCC.Core.GProp import GProp_GProps
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer, topexp_MapShapesAndAncestors
from OCC.Core.TopLoc import TopLoc_Location
from OCC.Core.TopoDS import topods
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListIteratorOfListOfShape,
)

# ==================== Constants ====================
# Маппинг типов признаков в числовые идентификаторы для единого вектора признаков GNN
TYPE_TO_ID = {
    'plane': 0,
    'cylinder': 1,
    'through_hole': 2,
    'blind_hole': 2,
    'pocket': 3,
    'slot': 4,
    'cone': 5,
    'sphere': 6,
    'chamfer': 7,
    'unknown': 8
}


# ==================== B-Rep Analysis ====================

def load_step(step_file: str):
    """
    Загружает 3D-модель из STEP-файла с помощью PythonOCC.
    
    Args:
        step_file: Путь к STEP-файлу.
        
    Returns:
        TopoDS_Shape: Корневая форма модели.
        
    Raises:
        RuntimeError: Если файл не может быть прочитан.
        ValueError: Если загруженная форма пуста.
    """
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    if status != IFSelect_RetDone:
        raise RuntimeError(f"Cannot load STEP file: {step_file}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape.IsNull():
        raise ValueError("Loaded shape is null")
    
    return shape


def extract_faces(shape) -> List[Dict]:
    """
    Извлекает все грани (Faces) из TopoDS_Shape и определяет их тип поверхности.
    
    Args:
        shape: TopoDS_Shape для анализа.
        
    Returns:
        List[Dict]: Список словарей с информацией о каждой грани (ID, объект, тип поверхности).
    """
    faces = []
    explorer = TopExp_Explorer(shape, TopAbs_FACE)
    face_id = 0
    
    while explorer.More():
        face = topods.Face(explorer.Current())
        surf = BRepAdaptor_Surface(face)
        faces.append({
            "face_id": face_id,
            "face": face,
            "surface_type": surf.GetType()
        })
        face_id += 1
        explorer.Next()
        
    return faces


def build_face_adjacency(shape, faces: List[Dict]) -> Dict[int, Set[int]]:
    """
    Строит карту смежности граней на основе общих ребер (Edges).
    
    Args:
        shape: TopoDS_Shape модели.
        faces: Список граней, полученных из extract_faces.
        
    Returns:
        Dict[int, Set[int]]: Словарь, где ключ - ID грани, значение - множество ID смежных граней.
    """
    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    topexp_MapShapesAndAncestors(shape, TopAbs_EDGE, TopAbs_FACE, edge_map)
    
    adjacency = {f["face_id"]: set() for f in faces}
    
    for i in range(1, edge_map.Size() + 1):
        face_list = edge_map.FindFromIndex(i)
        ids = []
        it = TopTools_ListIteratorOfListOfShape(face_list)
        
        while it.More():
            face = topods.Face(it.Value())
            for f in faces:
                if f["face"].IsSame(face):
                    ids.append(f["face_id"])
                    break
            it.Next()
            
        # Ребро соединяет ровно две грани
        if len(ids) == 2:
            a, b = ids
            adjacency[a].add(b)
            adjacency[b].add(a)
            
    return adjacency


# ==================== Rule-Based Feature Recognition ====================

def recognize_features_topology(faces: List[Dict], adjacency: Dict[int, Set[int]]) -> List[Dict]:
    """
    Иерархическое распознавание признаков на основе топологии граней.
    Порядок проверки важен: от наиболее специфичных (сквозные отверстия) к общим (плоскости).
    
    Args:
        faces: Список граней модели.
        adjacency: Карта смежности граней.
        
    Returns:
        List[Dict]: Список распознанных признаков с их типами и ID входящих в них граней.
    """
    face_types = {f["face_id"]: f["surface_type"] for f in faces}
    unassigned = set(face_types.keys())
    features = []

    # 1. HOLES (Цилиндр + >= 2 смежные плоскости = сквозное, 1 плоскость = глухое)
    for fid in list(unassigned):
        if face_types[fid] == GeomAbs_Cylinder:
            adj_planes = [n for n in adjacency[fid] if face_types[n] == GeomAbs_Plane]
            if len(adj_planes) >= 2:
                features.append({"type": "through_hole", "face_ids": [fid] + adj_planes})
                unassigned.difference_update([fid] + adj_planes)
            elif len(adj_planes) == 1:
                features.append({"type": "blind_hole", "face_ids": [fid] + adj_planes})
                unassigned.difference_update([fid] + adj_planes)

    # 2. POCKETS (Плоское дно + >= 3 смежные не-плоскости)
    for fid in list(unassigned):
        if face_types[fid] == GeomAbs_Plane:
            adj_non_planes = [n for n in adjacency[fid] if face_types[n] != GeomAbs_Plane]
            if len(adj_non_planes) >= 3:
                features.append({"type": "pocket", "face_ids": [fid] + adj_non_planes})
                unassigned.difference_update([fid] + adj_non_planes)

    # 3. SLOTS (Вытянутая группа плоскостей/цилиндров)
    # Эвристика: невыделенные плоскости, у которых ровно 2 соседние плоскости
    for fid in list(unassigned):
        if face_types[fid] == GeomAbs_Plane:
            adj_planes = [n for n in adjacency[fid] if n in unassigned and face_types[n] == GeomAbs_Plane]
            if len(adj_planes) == 2:
                group_faces = [fid] + adj_planes
                features.append({"type": "slot", "face_ids": group_faces})
                unassigned.difference_update(group_faces)

    # 4. CHAMFERS (Малая плоскость ровно с 2 смежными плоскостями)
    for fid in list(unassigned):
        if face_types[fid] == GeomAbs_Plane:
            adj_planes = [n for n in adjacency[fid] if n in unassigned and face_types[n] == GeomAbs_Plane]
            if len(adj_planes) == 2:
                features.append({"type": "chamfer", "face_ids": [fid]})
                unassigned.discard(fid)

    # 5. REMAINING PRIMITIVES (Одиночные невыделенные грани)
    for fid in list(unassigned):
        ftype = face_types[fid]
        if ftype == GeomAbs_Cylinder:
            features.append({"type": "cylinder", "face_ids": [fid]})
        elif ftype == GeomAbs_Cone:
            features.append({"type": "cone", "face_ids": [fid]})
        elif ftype == GeomAbs_Sphere:
            features.append({"type": "sphere", "face_ids": [fid]})
        elif ftype == GeomAbs_Plane:
            features.append({"type": "plane", "face_ids": [fid]})
        else:
            features.append({"type": "unknown", "face_ids": [fid]})
        unassigned.discard(fid)

    return features


# ==================== Point Cloud Mapping & Geometry ====================

def _estimate_cylinder_diameter(points: np.ndarray) -> float:
    """
    Оценивает диаметр цилиндра по облаку точек с помощью PCA.
    
    Args:
        points: Массив точек (N, 3).
        
    Returns:
        float: Оценочный диаметр цилиндра.
    """
    centroid = points.mean(axis=0)
    centered = points - centroid
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    
    # Ось цилиндра - направление максимальной дисперсии
    axis = eigvecs[:, np.argmax(eigvals)]
    proj_matrix = np.eye(3) - np.outer(axis, axis)
    projected = centered @ proj_matrix.T
    radii = np.linalg.norm(projected, axis=1)
    
    return float(2 * np.mean(radii))


def compute_curvature_pca(points: np.ndarray, k: int = 15) -> np.ndarray:
    """
    Вычисляет кривизну облака точек через локальный PCA.
    Формула: κ = λ_min / (λ_0 + λ_1 + λ_2)
    
    Args:
        points: Массив точек (N, 3).
        k: Количество ближайших соседей для локального анализа.
        
    Returns:
        np.ndarray: Массив значений кривизны для каждой точки.
    """
    tree = KDTree(points)
    curvatures = np.zeros(len(points))
    
    for i, p in enumerate(points):
        _, idx = tree.query(p, k=min(k, len(points)))
        neigh = points[idx]
        if len(neigh) < 3:
            continue
            
        pca = PCA(n_components=3)
        pca.fit(neigh)
        
        # Собственные значения (дисперсии), сортируем по возрастанию
        eig = np.sort(pca.explained_variance_)
        # Кривизна: отношение наименьшей дисперсии к сумме всех
        curvatures[i] = eig[0] / (eig.sum() + 1e-12)
        
    return curvatures


def _estimate_cone_parameters(pts: np.ndarray) -> Dict[str, float]:
    """
    Оценивает параметры усеченного конуса по облаку точек сегмента.
    
    Args:
        pts: Массив точек конуса (N, 3).
        
    Returns:
        Dict: Словарь с ключами 'height', 'base_radius', 'top_radius'.
    """
    if len(pts) < 10:
        return {"height": 0.0, "base_radius": 0.0, "top_radius": 0.0}
        
    centroid = pts.mean(axis=0)
    
    # 1. Ось конуса через PCA (направление максимальной дисперсии)
    pca = PCA(n_components=3)
    pca.fit(pts)
    axis = pca.components_[0]
    
    # 2. Проекция точек на ось
    t = (pts - centroid) @ axis
    
    # 3. Высота конуса
    height = float(t.max() - t.min())
    
    # 4. Разделение на нижнее и верхнее сечения (отсекаем по 10% краев для устойчивости к шуму)
    t_min_thresh = np.percentile(t, 10)
    t_max_thresh = np.percentile(t, 90)
    bottom_pts = pts[t <= t_min_thresh]
    top_pts = pts[t >= t_max_thresh]
    
    def calc_radius(slice_pts):
        if len(slice_pts) < 3:
            return 0.0
        # Расстояние от точки до прямой (оси конуса)
        cross_prods = np.cross(slice_pts - centroid, axis)
        radii = np.linalg.norm(cross_prods, axis=1)
        return float(np.mean(radii))
        
    base_radius = calc_radius(bottom_pts)
    top_radius = calc_radius(top_pts)
    
    # Гарантируем единообразие: base_radius всегда >= top_radius
    if top_radius > base_radius:
        base_radius, top_radius = top_radius, base_radius
        
    return {
        "height": round(height, 3),
        "base_radius": round(base_radius, 3),
        "top_radius": round(top_radius, 3)
    }


# ==================== GNN Encoding & Topology ====================

def encode_node_features(ftype: str, measured_value: dict, geometry: dict, confidence: float, degree: int) -> List[float]:
    """
    Создает единый вектор признаков фиксированной длины (18) для GNN.
    
    Args:
        ftype: Тип признака (строка).
        measured_value: Словарь с измеренными геометрическими параметрами.
        geometry: Словарь с агрегированными геометрическими свойствами.
        confidence: Уверенность в классификации (0.0 - 1.0).
        degree: Степень узла в графе смежности.
        
    Returns:
        List[float]: Вектор из 18 числовых признаков.
    """
    features = [0.0] * 18
    features[0] = float(TYPE_TO_ID.get(ftype, 8))
    
    # Базовые размеры
    if 'radius' in measured_value:
        features[1] = float(measured_value['radius'])
        features[2] = float(measured_value['radius']) * 2.0
    if 'diameter' in measured_value:
        features[2] = float(measured_value['diameter'])
        features[1] = float(measured_value['diameter']) / 2.0
    if 'length' in measured_value:
        features[3] = float(measured_value['length'])
    if 'width' in measured_value:
        features[4] = float(measured_value['width'])
    if 'depth' in measured_value:
        features[5] = float(measured_value['depth'])
    if 'height' in measured_value:
        features[6] = float(measured_value['height'])
        
    # Специфичные размеры для конусов
    if 'base_radius' in measured_value:
        features[8] = float(measured_value['base_radius'])
    if 'top_radius' in measured_value:
        features[9] = float(measured_value['top_radius'])
        
    # Геометрические агрегаты
    features[10] = float(geometry.get('surface_area_mm2', 0.0))
    curv = geometry.get('curvature', {})
    features[11] = float(curv.get('mean', 0.0))
    features[12] = float(curv.get('std', 0.0))
    features[13] = float(curv.get('max', 0.0))
    features[14] = float(geometry.get('normal_std', 0.0))
    features[15] = float(degree)          # Степень узла
    features[16] = float(confidence)
    # features[17] зарезервирован
    
    return features


def compute_relation(centroid_A: List[float], normal_A: List[float],
                     centroid_B: List[float], normal_B: List[float]) -> str:
    """
    Вычисляет топологическое отношение между двумя сегментами.
    Эвристика: скалярное произведение нормали A на вектор (Центроид_B - Центроид_A).
    
    Returns:
        str: "convex" (выпуклость), "concave" (вогнутость) или "orthogonal" (ортогонально).
    """
    n_A = np.array(normal_A)
    c_A = np.array(centroid_A)
    c_B = np.array(centroid_B)
    
    vec_AB = c_B - c_A
    
    # Нормализуем вектор и нормаль для чистоты вычислений
    norm_n = np.linalg.norm(n_A)
    if norm_n > 1e-6:
        n_A = n_A / norm_n
        
    norm_vec = np.linalg.norm(vec_AB)
    if norm_vec > 1e-6:
        vec_AB = vec_AB / norm_vec
        
    dot_product = np.dot(n_A, vec_AB)
    
    # Пороги для устойчивости к шуму
    if dot_product > 0.2:
        return "convex"
    elif dot_product < -0.2:
        return "concave"
    else:
        return "orthogonal"


def build_feature_config(points: np.ndarray, face_ids: np.ndarray,
                         features: List[Dict], adjacency: Dict[int, set],
                         face_to_feature: Dict[int, int], faces: List[Dict]) -> Dict:
    """
    Сопоставляет топологические признаки с точками облака и вычисляет 
    геометрию и топологию для формирования конфигурации GNN.
    
    Returns:
        Dict: Словарь с количеством признаков и списком их полных описаний.
    """
    config = []
    
    # ==========================================
    # ЭТАП 1: Сбор базовой геометрической информации о каждом сегменте
    # ==========================================
    for idx, feat in enumerate(features):
        mask = np.isin(face_ids, feat["face_ids"])
        point_indices = np.where(mask)[0].tolist()
        
        if len(point_indices) < 10:
            continue
            
        pts = points[point_indices]
        centroid = pts.mean(axis=0).tolist()
        bbox_min, bbox_max = pts.min(axis=0), pts.max(axis=0)
        ftype = feat["type"]
        
        # --- Нормали и кривизна ---
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts)
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
        normals = np.asarray(pcd.normals)
        
        mean_normal = normals.mean(axis=0)
        norm = np.linalg.norm(mean_normal)
        mean_normal = (mean_normal / (norm + 1e-10)).tolist() if norm > 1e-10 else [0.0, 0.0, 1.0]
        normal_std = float(np.std(normals, axis=0).mean())
        curvature = compute_curvature_pca(pts, k=15)
        
        # --- Измеряемые значения ---
        measured_value_dict = {}
        target_face_id = feat["face_ids"][0]
        target_face_info = next((f for f in faces if f["face_id"] == target_face_id), None)
        
        if ftype in ["through_hole", "blind_hole", "cylinder"]:
            measured_value_dict = {
                "diameter": float(_estimate_cylinder_diameter(pts)), 
                "depth": float(bbox_max[2] - bbox_min[2])
            }
        elif ftype == "sphere":
            if target_face_info and BRepAdaptor_Surface(target_face_info["face"]).GetType() == GeomAbs_Sphere:
                sphere_geom = BRepAdaptor_Surface(target_face_info["face"]).Sphere()
                measured_value_dict = {"radius": float(sphere_geom.Radius())}
            else:
                measured_value_dict = {"max_dimension": float(np.max(bbox_max - bbox_min))}
        elif ftype == "cone":
            cone_params = _estimate_cone_parameters(pts)
            measured_value_dict = {
                "height": cone_params["height"], 
                "base_radius": cone_params["base_radius"], 
                "top_radius": cone_params["top_radius"]
            }
        elif ftype in ["pocket", "slot"]:
            measured_value_dict = {
                "length": float(bbox_max[0] - bbox_min[0]), 
                "width": float(bbox_max[1] - bbox_min[1]), 
                "depth": float(bbox_max[2] - bbox_min[2])
            }
        else:
            measured_value_dict = {"max_dimension": float(np.max(bbox_max - bbox_min))}
            
        # --- Площадь через OCC ---
        real_area_mm2 = 0.0
        for f_id in feat["face_ids"]:
            face_occ = next((f["face"] for f in faces if f["face_id"] == f_id), None)
            if face_occ:
                props = GProp_GProps()
                brepgprop_SurfaceProperties(face_occ, props)
                real_area_mm2 += props.Mass()
                
        confidence = 0.9 if ftype != "unknown" else 0.5
        
        # Собираем временный объект сегмента
        config.append({
            "_idx": idx,  # Временный индекс для внутреннего маппинга
            "id": f"seg_{idx:03d}",
            "type": ftype,
            "centroid": centroid,
            "mean_normal": mean_normal,
            "measured_value": measured_value_dict,
            "position_3d": centroid,
            "geometry": {
                "bounding_box_mm": [bbox_min.tolist(), bbox_max.tolist()],
                "surface_area_mm2": float(real_area_mm2),
                "normal_std": normal_std,
                "curvature": {
                    "mean": float(np.mean(curvature)),
                    "std": float(np.std(curvature)),
                    "max": float(np.max(curvature))
                }
            },
            "point_indices": point_indices,
            "confidence": confidence,
            "source_face_ids": feat["face_ids"]
        })

    # ==========================================
    # ЭТАП 2: Семантическая топология, вложенность и векторы признаков
    # ==========================================
    # Создаем маппинги для безопасного доступа к временным данным
    idx_to_config_item = {item["_idx"]: item for item in config}
    idx_to_centroid = {item["_idx"]: item["centroid"] for item in config}
    idx_to_mean_normal = {item["_idx"]: item["mean_normal"] for item in config}
    
    face_to_orig_idx = {}
    for item in config:
        for f_id in item["source_face_ids"]:
            face_to_orig_idx[f_id] = item["_idx"]
            
    for item in config:
        idx = item["_idx"]
        adj_relations = []
        contains = []
        inside = []
        
        curr_bbox_min = np.array(item["geometry"]["bounding_box_mm"][0])
        curr_bbox_max = np.array(item["geometry"]["bounding_box_mm"][1])
        curr_centroid = np.array(idx_to_centroid[idx])
        curr_area = item["geometry"]["surface_area_mm2"]
        
        for f_id in item["source_face_ids"]:
            for nb_f_id in adjacency.get(f_id, []):
                nb_idx = face_to_orig_idx.get(nb_f_id)
                if nb_idx is not None and nb_idx != idx:
                    if not any(rel["segment"] == f"seg_{nb_idx:03d}" for rel in adj_relations):
                        nb_item = idx_to_config_item[nb_idx]
                        
                        # ШАГ 2.1: Вычисление отношения на уровне КОНКРЕТНЫХ ГРАНЕЙ
                        mask_f = (face_ids == f_id)
                        pts_f = points[mask_f]
                        if len(pts_f) >= 10:
                            pcd_f = o3d.geometry.PointCloud()
                            pcd_f.points = o3d.utility.Vector3dVector(pts_f)
                            pcd_f.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
                            normals_f = np.asarray(pcd_f.normals)
                            mean_normal_f = normals_f.mean(axis=0)
                            norm_f = np.linalg.norm(mean_normal_f)
                            mean_normal_f = (mean_normal_f / (norm_f + 1e-10)).tolist() if norm_f > 1e-10 else [0.0, 0.0, 1.0]
                            centroid_f = pts_f.mean(axis=0).tolist()
                        else:
                            mean_normal_f = idx_to_mean_normal[idx]
                            centroid_f = idx_to_centroid[idx]
                            
                        mask_nb = (face_ids == nb_f_id)
                        pts_nb = points[mask_nb]
                        if len(pts_nb) >= 10:
                            pcd_nb = o3d.geometry.PointCloud()
                            pcd_nb.points = o3d.utility.Vector3dVector(pts_nb)
                            pcd_nb.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=15))
                            normals_nb = np.asarray(pcd_nb.normals)
                            mean_normal_nb = normals_nb.mean(axis=0)
                            norm_nb = np.linalg.norm(mean_normal_nb)
                            mean_normal_nb = (mean_normal_nb / (norm_nb + 1e-10)).tolist() if norm_nb > 1e-10 else [0.0, 0.0, 1.0]
                            centroid_nb = pts_nb.mean(axis=0).tolist()
                        else:
                            mean_normal_nb = idx_to_mean_normal[nb_idx]
                            centroid_nb = idx_to_centroid[nb_idx]
                            
                        relation = compute_relation(centroid_f, mean_normal_f, centroid_nb, mean_normal_nb)
                        adj_relations.append({
                            "segment": f"seg_{nb_idx:03d}",
                            "relation": relation
                        })
                        
                        # ШАГ 2.2: ПРОВЕРКА ВЛОЖЕННОСТИ (CONTAINMENT)
                        nb_centroid = np.array(idx_to_centroid[nb_idx])
                        nb_area = nb_item["geometry"]["surface_area_mm2"]
                        nb_bbox_min = np.array(nb_item["geometry"]["bounding_box_mm"][0])
                        nb_bbox_max = np.array(nb_item["geometry"]["bounding_box_mm"][1])
                        
                        # Если центроид соседа внутри нашего BBox, и он значительно меньше по площади
                        if np.all(nb_centroid >= curr_bbox_min) and np.all(nb_centroid <= curr_bbox_max):
                            if nb_area < curr_area * 0.8:
                                contains.append(f"seg_{nb_idx:03d}")
                                
                        # Если наш центроид внутри BBox соседа, и мы значительно меньше по площади
                        if np.all(curr_centroid >= nb_bbox_min) and np.all(curr_centroid <= nb_bbox_max):
                            if curr_area < nb_area * 0.8:
                                inside.append(f"seg_{nb_idx:03d}")
                                
        degree = len(adj_relations)
        
        # Формируем финальный вектор признаков (18 значений)
        item["node_features"] = encode_node_features(
            item["type"], item["measured_value"], item["geometry"], item["confidence"], degree
        )
        
        # Записываем полную топологию, включая вложенность
        item["topology"] = {
            "source_face_ids": item["source_face_ids"],
            "adjacent_segments": adj_relations,
            "contains": list(set(contains)),   # Убираем дубликаты
            "inside": list(set(inside))
        }
        
        # Очистка временных полей перед финальной сериализацией
        del item["_idx"]
        del item["centroid"]
        del item["mean_normal"]
        del item["source_face_ids"]
        
    return {"num_features": len(config), "features": config}


# ==================== Point Cloud Generation ====================

def get_face_triangles(face):
    """Извлекает треугольники триангуляции для конкретной грани OCC."""
    location = TopLoc_Location()
    triangulation = BRep_Tool.Triangulation(face, location)
    if triangulation is None:
        return []
        
    triangles = []
    for i in range(1, triangulation.NbTriangles() + 1):
        tri = triangulation.Triangle(i)
        n1, n2, n3 = tri.Get()
        p1 = np.array(triangulation.Node(n1).Coord())
        p2 = np.array(triangulation.Node(n2).Coord())
        p3 = np.array(triangulation.Node(n3).Coord())
        triangles.append((p1, p2, p3))
    return triangles


def triangle_area(p1, p2, p3) -> float:
    """Вычисляет площадь треугольника в 3D пространстве."""
    return 0.5 * np.linalg.norm(np.cross(p2 - p1, p3 - p1))


def sample_triangle(p1, p2, p3, n: int) -> np.ndarray:
    """
    Равномерно сэмплирует n точек внутри заданного треугольника.
    Использует метод барицентрических координат.
    """
    r1 = np.sqrt(np.random.rand(n))
    r2 = np.random.rand(n)
    return ((1.0 - r1)[:, None] * p1 +
            (r1 * (1.0 - r2))[:, None] * p2 +
            (r1 * r2)[:, None] * p3)


def sample_face(face, face_id: int, points_per_face: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует облако точек для одной грани, пропорционально площади её треугольников.
    """
    triangles = get_face_triangles(face)
    if not triangles:
        return None, None
        
    areas = np.array([triangle_area(*tri) for tri in triangles])
    total_area = areas.sum()
    face_points, face_labels = [], []
    
    for tri, area in zip(triangles, areas):
        # Минимум 5 точек на треугольник для сохранения топологии
        n_pts = max(5, int(points_per_face * area / total_area))
        pts = sample_triangle(*tri, n_pts)
        face_points.append(pts)
        face_labels.extend([face_id] * len(pts))
        
    return np.vstack(face_points), np.array(face_labels)


def generate_face_aware_pointcloud(shape, mesh_deflection: float = 0.1,
                                   points_per_face: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """
    Генерирует глобальное облако точек с привязкой каждой точки к ID исходной грани.
    """
    mesh = BRepMesh_IncrementalMesh(shape, mesh_deflection)
    mesh.Perform()
    faces = extract_faces(shape)
    all_points, all_face_ids = [], []
    
    for face_info in faces:
        pts, labels = sample_face(face_info["face"], face_info["face_id"], points_per_face)
        if pts is not None:
            all_points.append(pts)
            all_face_ids.append(labels)
            
    return np.vstack(all_points), np.concatenate(all_face_ids)


# ==================== Visualization & Utilities ====================

def save_html_view(points, labels, feature_types=None, filename="features.html"):
    """Сохраняет интерактивную 3D-визуализацию сегментов в HTML с помощью Plotly."""
    points = np.asarray(points)
    labels = np.asarray(labels)
    
    if feature_types is None:
        unique_labels = np.unique(labels)
        feature_types = {label: f'Class {label}' for label in unique_labels}
        
    color_map = {
        'plane': '#1f77b4', 'hole': '#d62728', 'through_hole': '#d62728', 'blind_hole': '#ff7f0e',
        'slot': '#ff7f0e', 'pocket': '#9467bd', 'chamfer': '#2ca02c', 'fillet': '#e377c2',
        'cylinder': '#17becf', 'sphere': '#8c564b', 'cone': '#bcbd22', 'unknown': '#7f7f7f',
    }
    
    traces = []
    unique_labels = np.unique(labels)
    for label in unique_labels:
        mask = labels == label
        if not np.any(mask):
            continue
            
        feature_name = str(feature_types.get(label, f'Class {label}')).strip()
        color = color_map.get(feature_name.lower(), '#7f7f7f')
        
        trace = go.Scatter3d(
            x=points[mask, 0], y=points[mask, 1], z=points[mask, 2],
            mode='markers', name=feature_name,
            marker=dict(size=3, color=color, opacity=0.8, line=dict(color=color, width=0.5)),
            hoverinfo='name', legendgroup=feature_name
        )
        traces.append(trace)
        
    fig = go.Figure(data=traces)
    fig.update_layout(
        title=dict(text='3D Feature Segmentation', x=0.5, y=0.95),
        scene=dict(aspectmode='data'),
        showlegend=True,
        legend=dict(title=dict(text='Feature Types'))
    )
    fig.write_html(filename, include_plotlyjs='cdn', full_html=True)
    print(f"  Saved HTML visualization to: {filename}")


def visualize_features(points: np.ndarray, labels: np.ndarray, features, filename: str = "features_3d.html"):
    """Обертка для визуализации признаков с помощью Open3D и сохранения в HTML."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    labels = np.array(labels)
    
    cmap = plt.get_cmap("tab20")
    colors = cmap(labels % 20)[:, :3]
    pcd.colors = o3d.utility.Vector3dVector(colors)
    
    unique_labels = np.unique(labels)
    feature_types = {}
    for label in unique_labels:
        target_id = f"seg_{label:03d}"
        matched = False
        for feat in features:
            feat_id = str(feat.get('id', '')).strip()
            if feat_id == target_id:
                feature_types[label] = str(feat.get('type', f'Class {label}')).strip()
                matched = True
                break
        if not matched:
            feature_types[label] = f'Class {label}'
            
    # Передаем filename в save_html_view
    save_html_view(points, labels, feature_types, filename=filename)


def save_pointcloud(points: np.ndarray, filename: str):
    """Сохраняет облако точек в формате PLY."""
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    o3d.io.write_point_cloud(filename, pcd)


def save_config(config: Dict, filename: str = "features_config.json"):
    """Сохраняет конфигурацию признаков в JSON."""
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {filename}")


def build_point_feature_labels(face_ids: np.ndarray, face_to_feature: Dict[int, int]) -> np.ndarray:
    """Создает массив меток признаков для каждой точки облака на основе ID грани."""
    return np.array([face_to_feature.get(fid, -1) for fid in face_ids])


# ==================== MAIN Execution ====================

def main():
    """Основной конвейер обработки STEP-файла."""
    STEP_FILE = "data/abc_dataset/abc_0000_step_v00/00000001/00000001_1ffb81a71e5b402e966b9341_step_000.step"
    TARGET_POINTS = 4096
    
    print(f"Loading STEP: {STEP_FILE}")
    shape = load_step(STEP_FILE)
    
    print("Recognizing features from B-Rep topology...")
    faces = extract_faces(shape)
    adjacency = build_face_adjacency(shape, faces)
    features_topology = recognize_features_topology(faces, adjacency)
    print(f"Found {len(features_topology)} topological features")
    
    print("Generating face-aware point cloud...")
    # points_per_face=500 дает ~5000-15000 точек для типичной детали, что оптимально
    points, face_ids = generate_face_aware_pointcloud(shape, mesh_deflection=0.1, points_per_face=500)
    print(f"Generated {len(points)} points with face mapping")
    
    # Построение маппинга: ID грани -> ID признака
    face_to_feature = {}
    for feat_idx, feat in enumerate(features_topology):
        for fid in feat["face_ids"]:
            face_to_feature[fid] = feat_idx
            
    feature_labels = build_point_feature_labels(face_ids, face_to_feature)
    n_segments = len(np.unique(feature_labels[feature_labels >= 0]))
    print(f"Mapped into {n_segments} segments")
    
    print("Mapping topology to point cloud...")
    config = build_feature_config(points, face_ids, features_topology, adjacency, face_to_feature, faces)
    
    print("Visualizing and saving...")
    save_pointcloud(points, "pointcloud.ply")
    
    # Подготовка меток для визуализации
    labels = np.full(len(points), -1, dtype=int)
    for idx, feat in enumerate(config["features"]):
        labels[feat["point_indices"]] = idx
        
    visualize_features(points, labels, config["features"])
    save_config(config, "features_config.json")
    
    print("\nFeature statistics:")
    for feat in config["features"]:
        print(f"  {feat['id']}: {feat['type']} ({feat['measured_value']} mm, {len(feat['point_indices'])} pts)")
        
    print("\nDone! Output files:")
    print("  • pointcloud.ply")
    print("  • features_config.json")
    print("  • features_3d.html")


if __name__ == "__main__":
    main()