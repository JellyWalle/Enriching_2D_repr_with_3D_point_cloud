#!/usr/bin/env python3

import json

import numpy as np
import open3d as o3d

from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.IFSelect import IFSelect_RetDone

from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopAbs import TopAbs_FACE

from OCC.Core.TopoDS import topods
from OCC.Core.TopLoc import TopLoc_Location

from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.BRep import BRep_Tool

from OCC.Core.TopAbs import (
    TopAbs_FACE,
    TopAbs_EDGE
)

from OCC.Core.TopExp import (
    topexp_MapShapesAndAncestors
)

from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape
)

from OCC.Core.BRepAdaptor import (
    BRepAdaptor_Surface
)

from OCC.Core.GeomAbs import (
    GeomAbs_Plane,
    GeomAbs_Cylinder
)

import numpy as np
from OCC.Core.BRepGProp import brepgprop_SurfaceProperties
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
from OCC.Core.GeomAbs import *

def analyze_feature(feature_faces):
    types = []
    areas = []

    for f in feature_faces:
        surf = BRepAdaptor_Surface(f["face"])

        types.append(surf.GetType())

        props = GProp_GProps()
        brepgprop_SurfaceProperties(f["face"], props)
        areas.append(props.Mass())

    types = np.array(types)
    areas = np.array(areas)

    return {
        "dominant_type": np.bincount(types).argmax(),
        "area": areas.sum(),
        "type_hist": np.bincount(types, minlength=10),
        "mixed": len(set(types)) > 1
    }

def classify_feature(desc):

    if desc["dominant_type"] == GeomAbs_Cylinder:
        return "hole_or_boss"

    if desc["dominant_type"] == GeomAbs_Torus:
        if desc["area"] < 100:
            return "fillet"
        else:
            return "revolved_surface"

    if desc["dominant_type"] == GeomAbs_Plane:
        return "plane_patch"

    if desc["dominant_type"] == GeomAbs_BSplineSurface:
        return "freeform"

    return "unknown"

def extract_faces(shape):

    faces = []

    explorer = TopExp_Explorer(
        shape,
        TopAbs_FACE
    )

    face_id = 0

    while explorer.More():

        face = topods.Face(
            explorer.Current()
        )

        surf = BRepAdaptor_Surface(face)

        faces.append({
            "face_id": face_id,
            "face": face,
            "surface_type": surf.GetType()
        })

        face_id += 1

        explorer.Next()

    return faces
from OCC.Core.TopExp import topexp
from OCC.Core.TopTools import (
    TopTools_IndexedDataMapOfShapeListOfShape,
    TopTools_ListIteratorOfListOfShape
)
from OCC.Core.TopAbs import (
    TopAbs_EDGE,
    TopAbs_FACE
)
from OCC.Core.TopoDS import topods


def build_face_adjacency(shape, faces):

    edge_map = (
        TopTools_IndexedDataMapOfShapeListOfShape()
    )

    topexp.MapShapesAndAncestors(
        shape,
        TopAbs_EDGE,
        TopAbs_FACE,
        edge_map
    )

    adjacency = {
        f["face_id"]: set()
        for f in faces
    }

    for i in range(
        1,
        edge_map.Size() + 1
    ):

        face_list = edge_map.FindFromIndex(i)

        ids = []

        it = TopTools_ListIteratorOfListOfShape(
            face_list
        )

        while it.More():

            face = topods.Face(
                it.Value()
            )

            for f in faces:

                if f["face"].IsSame(face):

                    ids.append(
                        f["face_id"]
                    )

                    break

            it.Next()

        if len(ids) == 2:

            a, b = ids

            adjacency[a].add(b)
            adjacency[b].add(a)

    return adjacency


from collections import deque

def get_connected_components(adjacency):
    visited = set()
    components = []

    for node in adjacency:
        if node in visited:
            continue

        comp = []
        q = deque([node])
        visited.add(node)

        while q:
            n = q.popleft()
            comp.append(n)

            for nb in adjacency[n]:
                if nb not in visited:
                    visited.add(nb)
                    q.append(nb)

        components.append(comp)

    return components

def build_face_to_feature_map(components):
    face_to_feature = {}

    for fid, comp in enumerate(components):
        for f in comp:
            face_to_feature[f] = fid

    return face_to_feature

def build_point_feature_labels(face_ids, face_to_feature):
    return np.array([
        face_to_feature[fid]
        for fid in face_ids
    ])
import matplotlib.pyplot as plt

import plotly.graph_objects as go
import numpy as np

def save_html_view(points, labels, filename="segmentation.html"):

    points = np.asarray(points)
    labels = np.asarray(labels)

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(
                    size=1,
                    color=labels,
                    colorscale="Turbo",
                    opacity=0.9
                )
            )
        ]
    )

    fig.update_layout(
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z"
        ),
        title="Feature Segmentation"
    )

    fig.write_html(filename)

    print(f"Saved HTML: {filename}")

def visualize_features(points, labels):

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)

    labels = np.array(labels)
    n = labels.max() + 1 if len(labels) > 0 else 1

    cmap = plt.get_cmap("tab20")
    colors = cmap(labels % 20)[:, :3]

    pcd.colors = o3d.utility.Vector3dVector(colors)

    #o3d.visualization.draw_geometries([pcd])
    save_html_view(points, feature_labels, "features.html")
    
def detect_holes(
        faces,
        adjacency):

    face_map = {
        f["face_id"]: f
        for f in faces
    }

    holes = []

    for f in faces:

        if (
            f["surface_type"]
            != GeomAbs_Cylinder
        ):
            continue

        fid = f["face_id"]

        plane_neighbors = []

        for nb in adjacency[fid]:

            if (
                face_map[nb]["surface_type"]
                == GeomAbs_Plane
            ):
                plane_neighbors.append(
                    nb
                )

        if len(
            plane_neighbors
        ) >= 2:

            holes.append({

                "type": "hole",

                "faces": [
                    fid,
                    *plane_neighbors
                ]
            })

    return holes


def recognize_features(
        shape):

    faces = extract_faces(
        shape
    )

    adjacency = (
        build_face_adjacency(
            shape,
            faces
        )
    )

    holes = detect_holes(
        faces,
        adjacency
    )

    return {
        "faces": faces,
        "adjacency": adjacency,
        "holes": holes
    }


# ============================================================
# STEP LOADER
# ============================================================

def load_step(step_file):

    reader = STEPControl_Reader()

    status = reader.ReadFile(step_file)

    if status != IFSelect_RetDone:
        raise RuntimeError(f"Cannot load STEP file: {step_file}")

    reader.TransferRoots()

    shape = reader.OneShape()

    return shape


# ============================================================
# FACE EXTRACTION
# ============================================================
'''
def extract_faces(shape):

    faces = []

    explorer = TopExp_Explorer(shape, TopAbs_FACE)

    face_id = 0

    while explorer.More():

        face = topods.Face(explorer.Current())

        faces.append(
            {
                "face_id": face_id,
                "face": face
            }
        )

        face_id += 1

        explorer.Next()

    return faces

'''


# ============================================================
# TRIANGLES
# ============================================================

from OCC.Core.BRep import BRep_Tool
from OCC.Core.TopLoc import TopLoc_Location
import numpy as np


def get_face_triangles(face):

    location = TopLoc_Location()

    triangulation = BRep_Tool.Triangulation(
        face,
        location
    )

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


# ============================================================
# GEOMETRY
# ============================================================

def triangle_area(p1, p2, p3):

    return 0.5 * np.linalg.norm(
        np.cross(
            p2 - p1,
            p3 - p1
        )
    )


def sample_triangle(p1, p2, p3, n):

    r1 = np.sqrt(np.random.rand(n))
    r2 = np.random.rand(n)

    points = (
        (1.0 - r1)[:, None] * p1
        + (r1 * (1.0 - r2))[:, None] * p2
        + (r1 * r2)[:, None] * p3
    )

    return points


# ============================================================
# FACE SAMPLING
# ============================================================

def sample_face(
        face,
        face_id,
        points_per_face=1000):

    triangles = get_face_triangles(face)

    if len(triangles) == 0:
        return None, None

    areas = np.array([
        triangle_area(*tri)
        for tri in triangles
    ])

    total_area = np.sum(areas)

    face_points = []
    face_labels = []

    for tri, area in zip(triangles, areas):

        n_points = max(
            1,
            int(points_per_face * area / total_area)
        )

        pts = sample_triangle(*tri, n_points)

        face_points.append(pts)

        face_labels.extend(
            [face_id] * len(pts)
        )

    face_points = np.vstack(face_points)

    return face_points, np.array(face_labels)


# ============================================================
# POINT CLOUD GENERATION
# ============================================================

def generate_face_aware_pointcloud(
        shape,
        mesh_deflection=0.1,
        points_per_face=1000):

    print("Meshing model...")

    mesh = BRepMesh_IncrementalMesh(
        shape,
        mesh_deflection
    )

    mesh.Perform()

    faces = extract_faces(shape)

    print(f"Found faces: {len(faces)}")
    

    all_points = []
    all_face_ids = []

    for face_info in faces:

        face_id = face_info["face_id"]
        face = face_info["face"]
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        surf = BRepAdaptor_Surface(face)
        print(
            face_info["face_id"],
            surf.GetType()
        )

        pts, labels = sample_face(
            face_info["face"],
            face_id,
            points_per_face
        )

        if pts is None:
            continue

        all_points.append(pts)
        all_face_ids.append(labels)

    points = np.vstack(all_points)

    face_ids = np.concatenate(all_face_ids)

    return points, face_ids


# ============================================================
# SAVE
# ============================================================

def save_pointcloud(points, filename):

    pcd = o3d.geometry.PointCloud()

    pcd.points = o3d.utility.Vector3dVector(points)

    o3d.io.write_point_cloud(
        filename,
        pcd
    )

def cylinder_radius(face):

    surf = BRepAdaptor_Surface(
        face
    )

    cyl = surf.Cylinder()

    return cyl.Radius()

# ============================================================
# MAIN
# ============================================================

from OCC.Core.BRepAdaptor import BRepAdaptor_Surface


def dump_cylinders(faces):

    for f in faces:

        if f["surface_type"] != GeomAbs_Cylinder:
            continue

        surf = BRepAdaptor_Surface(
            f["face"]
        )

        cyl = surf.Cylinder()

        print(
            f"FACE {f['face_id']}"
        )

        print(
            "radius =",
            cyl.Radius()
        )

        print(
            "location =",
            cyl.Location().Coord()
        )

        print()
'''
def build_feature_config(
    points,
    face_ids,
    feature_labels,
    features_info=None
):

    config = {
        "num_features": int(len(np.unique(feature_labels))),
        "features": []
    }

    unique_features = np.unique(feature_labels)

    for fid in unique_features:

        mask = feature_labels == fid
        pts = points[mask]

        face_mask = face_ids[mask]

        feature = {
            "id": f"feature_{fid:03d}",
            "type": "geometry",
            "feature_type": "unknown",
            "num_points": int(len(pts)),
            "position_3d": np.mean(pts, axis=0).tolist(),
            "bounding_box": [
                pts.min(axis=0).tolist(),
                pts.max(axis=0).tolist()
            ],
            "point_indices": np.where(mask)[0].tolist(),
            "confidence": 0.85
        }

        config["features"].append(feature)

    return config
'''
def save_config(config, filename="features_config.json"):
    with open(filename, "w") as f:
        json.dump(config, f, indent=2)
    print(f"Saved config: {filename}")

def downsample_points(points, face_ids, max_points=1024):

    points = np.asarray(points)
    face_ids = np.asarray(face_ids)

    n = len(points)

    if n <= max_points:
        return points, face_ids

    idx = np.random.choice(n, max_points, replace=False)

    return points[idx], face_ids[idx]

def classify_feature_from_segment(pts, face_types, adjacency_size):
    """
    pts - (N,3)
    face_types - list of OCC surface types
    adjacency_size - int
    """

    unique_types = set(face_types)
    n = len(pts)

    # ---- HOLE ----
    # цилиндр + окружён плоскостями
    if GeomAbs_Cylinder in unique_types:
        if adjacency_size >= 2:
            return "hole"

    # ---- POCKET ----
    # замкнутые области на плоскости
    if unique_types == {GeomAbs_Plane}:
        if n > 200:
            return "pocket"
        else:
            return "planar_face"

    # ---- SLOT ----
    # смесь цилиндров + плоскостей + вытянутая форма
    if GeomAbs_Cylinder in unique_types and GeomAbs_Plane in unique_types:
        return "slot"

    # ---- CHAMFER ----
    if GeomAbs_Plane in unique_types and n < 200:
        return "chamfer"

    # ---- FILLET (bonus) ----
    if GeomAbs_Torus in unique_types or GeomAbs_Cone in unique_types:
        return "fillet"

    return "unknown"

def normalize_feature_labels(feature_labels):
    """
    Приводит feature_labels к виду:
    np.ndarray[int]
    """

    if isinstance(feature_labels, np.ndarray):
        return feature_labels.astype(int)

    if isinstance(feature_labels, list):
        out = []
        for f in feature_labels:
            if isinstance(f, dict):
                # если вдруг dict → достаём id
                if "feature_id" in f:
                    out.append(f["feature_id"])
                elif "label" in f:
                    out.append(f["label"])
                else:
                    raise ValueError(f"Unknown dict format: {f}")
            else:
                out.append(int(f))
        return np.array(out)

    return np.array(feature_labels, dtype=int)
def build_feature_config(points, faces, feature_labels):

    feature_labels = normalize_feature_labels(feature_labels)

    from collections import defaultdict

    feature_points = defaultdict(list)

    for i, fid in enumerate(feature_labels):
        feature_points[fid].append(points[i])

    config = []

    for fid, pts in feature_points.items():

        pts = np.array(pts)

        min_bb = pts.min(axis=0)
        max_bb = pts.max(axis=0)

        centroid = pts.mean(axis=0)

        config.append({
            "id": f"seg_{fid}",
            "type": "unknown",  # пока без классификации
            "measured_value": len(pts),
            "position_3d": centroid.tolist(),
            "geometry": {
                "bounding_box": [min_bb.tolist(), max_bb.tolist()]
            },
            "topology": {
                "surface_area": len(pts)
            }
        })

    return {
        "num_features": len(config),
        "features": config
    }


if __name__ == "__main__":

    STEP_FILE = "data/abc_dataset/abc_0000_step_v00/00000000/00000000_290a9120f9f249a7a05cfe9c_step_000.step"

    from OCC.Core.TopExp import *
    from OCC.Core.TopTools import *

    print("MapShapesAndAncestors" in dir())
    print("TopTools_IndexedDataMapOfShapeListOfShape" in dir())
    edge_map = TopTools_IndexedDataMapOfShapeListOfShape()
    print(type(edge_map))
    print(dir(edge_map))

    print()

    from OCC.Core.GeomAbs import *

    print("Plane =", int(GeomAbs_Plane))
    print("Cylinder =", int(GeomAbs_Cylinder))
    print("Cone =", int(GeomAbs_Cone))
    print("Sphere =", int(GeomAbs_Sphere))
    print("Torus =", int(GeomAbs_Torus))
    print("BSpline =", int(GeomAbs_BSplineSurface))

    import OCC.Core.TopExp as TE
    import OCC.Core.TopTools as TT

    print([x for x in dir(TE) if "Ancestor" in x])
    print([x for x in dir(TT) if "IndexedDataMap" in x])

    shape = load_step(STEP_FILE)
    
    points, face_ids = generate_face_aware_pointcloud(
        shape,
        mesh_deflection=0.1,
        points_per_face=1000
    )

    points, face_ids = downsample_points(points, face_ids, 1024)

    print()
    print("POINT CLOUD GENERATED")
    print("points:", len(points))
    print("unique faces:", len(np.unique(face_ids)))

    features = recognize_features(shape)

    components = get_connected_components(features["adjacency"])
    print("Features (connected components):", len(components))

    face_to_feature = build_face_to_feature_map(components)

    feature_labels = build_point_feature_labels(face_ids, face_to_feature)
    

    print("Feature labels:", len(np.unique(feature_labels)))
    save_pointcloud(points, "pointcloud.ply")
    visualize_features(points, feature_labels)

    unique_features = np.unique(feature_labels)

    print("\nFeature statistics:")

    for fid in unique_features:
        count = np.sum(feature_labels == fid)
        print(f"Feature {fid:4d} -> {count:8d} points")
    #exit(0)

    dump_cylinders(
        features["faces"]
    )

    print("\nFACE GRAPH")

    for fid, neigh in features["adjacency"].items():

        print(
            fid,
            sorted(list(neigh))
        )

    print()
    print("HOLES FOUND")

    for i, hole in enumerate(features["holes"]):
        print(
            f"Hole {i}: "
            f"{hole['faces']}"
        )
        cyl_face = hole["faces"][0]
        face = (
            features["faces"]
            [cyl_face]
            ["face"]
        )

        r = cylinder_radius(
            face
        )

        print(
            f"Hole {i}"
            f" diameter = "
            f"{2*r:.3f}"
        )

    feature_labels = build_point_feature_labels(face_ids, face_to_feature)
    config = build_feature_config(
        points,
        features["faces"],
        feature_labels
    )

    save_config(config, "features_config.json")
    np.save(
        "face_ids.npy",
        face_ids
    )

    save_pointcloud(
        points,
        "pointcloud.ply"
    )

    unique_faces = np.unique(face_ids)

    print()
    print("Face statistics:")

    for fid in unique_faces:

        count = np.sum(face_ids == fid)

        print(
            f"Face {fid:4d} -> {count:8d} points"
        )

    print()
    print("Saved:")
    print("  points.npy")
    print("  face_ids.npy")
    print("  pointcloud.ply")