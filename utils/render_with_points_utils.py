#!/usr/bin/env python3
"""
Генератор инженерных чертежей из STEP-файлов с допусками.

Решает проблемы:
1. Консистентность допусков через ToleranceManager (кэширование)
2. Правильное масштабирование и размещение чертежа
3. Извлечение реальных размеров из геометрии
4. Размещение размерных линий без пересечения с геометрией
"""

import os
import numpy as np
import svgwrite
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.GeomAbs import GeomAbs_Line, GeomAbs_Circle, GeomAbs_Ellipse, GeomAbs_C1
from OCC.Core.gp import gp_Pnt, gp_Dir, gp_Ax2, gp_Vec, gp_Pnt2d
from OCC.Core.HLRAlgo import HLRAlgo_Projector
from OCC.Core.HLRBRep import HLRBRep_Algo, HLRBRep_HLRToShape
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_Reader
from OCC.Core.TopAbs import TopAbs_EDGE, TopAbs_FACE
from OCC.Core.TopExp import TopExp_Explorer
from OCC.Core.TopoDS import topods
from OCC.Core.Bnd import Bnd_Box2d
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.GeomAPI import GeomAPI_PointsToBSpline
from OCC.Core.GeomConvert import GeomConvert_BSplineCurveToBezierCurve


# ==================== Менеджер допусков (Единый источник истины) ====================
class ToleranceManager:
    """
    Менеджер допусков, гарантирующий консистентность размеров across всех видов.
    Генерирует допуск ОДИН раз для каждого уникального номинального размера.
    """
    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)
        self.tolerance_cache = {}  # Кэш: nominal_value -> tolerance_dict
        
    def get_tolerance(self, nominal_value: float) -> Dict:
        """
        Получает допуск для номинального значения.
        Если допуск уже был сгенерирован для этого значения - возвращает кэшированный.
        """
        # Округляем до 2 знаков для ключа кэша
        cache_key = round(float(nominal_value), 2)
        
        if cache_key not in self.tolerance_cache:
            # Генерируем допуск ОДИН раз
            tolerance_type = self.rng.choice(['bilateral', 'unilateral_upper', 'unilateral_lower'])
            tolerance_value = nominal_value * self.rng.uniform(0.01, 0.03)  # 1-3%
            
            if tolerance_type == 'bilateral':
                self.tolerance_cache[cache_key] = {
                    'type': 'bilateral',
                    'value': tolerance_value,
                    'display': f'±{tolerance_value:.2f}',
                    'upper_limit': nominal_value + tolerance_value,
                    'lower_limit': nominal_value - tolerance_value
                }
            elif tolerance_type == 'unilateral_upper':
                self.tolerance_cache[cache_key] = {
                    'type': 'unilateral_upper',
                    'value': tolerance_value,
                    'display': f'+{tolerance_value:.2f}/0',
                    'upper_limit': nominal_value + tolerance_value,
                    'lower_limit': nominal_value
                }
            else:  # unilateral_lower
                self.tolerance_cache[cache_key] = {
                    'type': 'unilateral_lower',
                    'value': tolerance_value,
                    'display': f'0/-{tolerance_value:.2f}',
                    'upper_limit': nominal_value,
                    'lower_limit': nominal_value - tolerance_value
                }
        
        return self.tolerance_cache[cache_key]
    
    def clear_cache(self):
        """Очистить кэш (для нового STEP-файла)"""
        self.tolerance_cache.clear()


# ==================== HLR проекция ====================
def get_sorted_hlr_edges(shape, position=(0, 0, 0), direction=(1, 1, 1),
                         export_hidden_edges=False):
    """
    Получение отсортированных рёбер HLR.
    """
    if isinstance(position, gp_Pnt):
        location_gp = position
    else:
        location_gp = gp_Pnt(*position)
    
    if isinstance(direction, gp_Dir):
        direction_gp = direction
    else:
        direction_gp = gp_Dir(*direction)
    
    projector = HLRAlgo_Projector(gp_Ax2(location_gp, direction_gp))
    
    hlr_algo = HLRBRep_Algo()
    hlr_algo.Add(shape)
    hlr_algo.Projector(projector)
    hlr_algo.Update()
    hlr_algo.Hide()
    
    hlr_to_shape = HLRBRep_HLRToShape(hlr_algo)
    visible_edges = hlr_to_shape.VCompound()
    
    hidden_edges = None
    if export_hidden_edges:
        hidden_edges = hlr_to_shape.Rg1LineVCompound()
    
    return visible_edges, hidden_edges


def discretize_edge(edge, deflection=0.01):
    """Дискретизация ребра в точки."""
    curve = BRepAdaptor_Curve(edge)
    points = []
    u_min = curve.FirstParameter()
    u_max = curve.LastParameter()
    num_points = max(10, int((u_max - u_min) / deflection))
    
    for i in range(num_points + 1):
        u = u_min + (u_max - u_min) * i / num_points
        point = curve.Value(u)
        points.append(point.Coord()[:3])
    
    return np.array(points)


# ==================== Вспомогательные функции для SVG ====================
def _Tcol_dim_1(li, _type):
    """Function factory for 1-dimensional TCol* types."""
    pts = _type(0, len(li) - 1)
    for n, i in enumerate(li):
        pts.SetValue(n, i)
    return pts


def point_list_to_TColgp_Array1OfPnt(li):
    return _Tcol_dim_1(li, TColgp_Array1OfPnt)


def add_to_bounding_box(points_2d):
    box2d = Bnd_Box2d()
    for p in points_2d:
        box2d.Add(gp_Pnt2d(*p))
    return box2d


def line_to_svg(curve):
    line = curve.Line()
    location = np.array(line.Location().Coord()[:2])
    direction = np.array(line.Direction().Coord()[:2])
    start = location + curve.FirstParameter() * direction
    end = location + curve.LastParameter() * direction
    return svgwrite.shapes.Line(start, end, fill="none"), add_to_bounding_box((start, end))


def approx_points_by_piecewise_bezier(points_3d, degree, tol):
    if degree not in (2, 3):
        raise RuntimeError("SVG files only support Bezier curves of degree 2 or 3")
    
    points_3d_occ = [gp_Pnt(*p) for p in points_3d]
    approx_spline = GeomAPI_PointsToBSpline(
        point_list_to_TColgp_Array1OfPnt(points_3d_occ), degree, degree, GeomAbs_C1, tol
    )
    
    if not approx_spline.IsDone():
        raise RuntimeError("Could not approximate points within a given tolerance")
    
    return GeomConvert_BSplineCurveToBezierCurve(approx_spline.Curve())


def piecewise_bezier_to_svg(points_2d, bezier_curves, degree):
    path_elements = []
    
    for i in range(1, bezier_curves.NbArcs() + 1):
        if bezier_curves.Arc(i).Degree() != degree:
            raise RuntimeError(f"Approximated degree of Bezier curves is not {degree}")
        
        curve = bezier_curves.Arc(i)
        if degree == 2:
            start, control, end = [p.Coord()[:2] for p in list(curve.Poles())]
            path_elements.append(f"M {start[0]},{start[1]} Q {control[0]},{control[1]} {end[0]},{end[1]}".split())
        elif degree == 3:
            start, first_control, second_control, end = [p.Coord()[:2] for p in list(curve.Poles())]
            path_elements.append(
                f"M {start[0]},{start[1]} C {first_control[0]},{first_control[1]} {second_control[0]},{second_control[1]} {end[0]},{end[1]}".split()
            )
        path_elements.append(f"M{end[0]} {end[1]}".split())
    
    return svgwrite.path.Path(d=path_elements, fill="none"), add_to_bounding_box(points_2d)


def polyline_to_svg(points_2d):
    return svgwrite.shapes.Polyline(points_2d, fill="none"), add_to_bounding_box(points_2d)


CURVE_TYPES = {0: 'Line', 1: 'Circle', 2: 'Ellipse', 6: 'BSpline'}


def edge_to_svg(topods_edge, bezier_tol=0.01, bezier_degree=2):
    """Returns a svgwrite.Path for the edge, and the 2d bounding box."""
    curve = BRepAdaptor_Curve(topods_edge)
    
    if curve.GetType() == 0:  # line
        return line_to_svg(curve)
    else:
        points_3d = discretize_edge(topods_edge, deflection=0.01)
        points_2d = [p[:2] for p in points_3d]
        
        try:
            return piecewise_bezier_to_svg(
                points_2d, 
                approx_points_by_piecewise_bezier(points_3d, bezier_degree, bezier_tol), 
                bezier_degree
            )
        except RuntimeError:
            print(f"Converting {CURVE_TYPES.get(curve.GetType(), 'Unknown')} to polyline")
            return polyline_to_svg(points_2d)


# ==================== Основная функция генерации чертежа ====================
def export_shape_to_svg_with_tolerances(
    shape, filename=None,
    width=800, height=600,
    margin_left=50, margin_top=50,
    margin_right=50, margin_bottom=80,
    export_hidden_edges=False,
    location=(0, 0, 0), direction=(1, 1, 1),
    bezier_degree=2, bezier_tol=0.01,
    color="black", line_width=0.5,
    add_tolerances=True,
    tolerance_manager=None
):
    """Экспорт STEP-формы в SVG с читаемыми размерными линиями."""
    if shape.IsNull():
        raise AssertionError("shape is Null")
    
    # Получаем рёбра HLR
    location_gp = gp_Pnt(*location)
    direction_gp = gp_Dir(*direction)
    visible_edges, hidden_edges = get_sorted_hlr_edges(
        shape, position=location_gp, direction=direction_gp,
        export_hidden_edges=export_hidden_edges
    )
    
    # Собираем пути и вычисляем bounding box
    global_2d_bounding_box = Bnd_Box2d()
    paths = []
    
    # === ИСПРАВЛЕНИЕ: Итерируемся через TopExp_Explorer ===
    if visible_edges and not visible_edges.IsNull():
        explorer = TopExp_Explorer(visible_edges, TopAbs_EDGE)
        while explorer.More():
            edge = topods.Edge(explorer.Current())
            path, box = edge_to_svg(edge, bezier_tol=bezier_tol, bezier_degree=bezier_degree)
            if path:
                paths.append(path)
                global_2d_bounding_box.Add(box)
            explorer.Next()
            
    if export_hidden_edges and hidden_edges and not hidden_edges.IsNull():
        explorer = TopExp_Explorer(hidden_edges, TopAbs_EDGE)
        while explorer.More():
            edge = topods.Edge(explorer.Current())
            path, box = edge_to_svg(edge, bezier_tol=bezier_tol, bezier_degree=bezier_degree)
            if path:
                path.dasharray([5, 5])
                paths.append(path)
                global_2d_bounding_box.Add(box)
            explorer.Next()
    # ====================================================
    
    if global_2d_bounding_box.IsVoid():
        print("Warning: No visible edges in projection.")
        return None
        
    x_min, y_min, x_max, y_max = global_2d_bounding_box.Get()
    bb2d_width = x_max - x_min
    bb2d_height = y_max - y_min
    center_x = (x_min + x_max) / 2
    center_y = (y_min + y_max) / 2
    
    # Создаем SVG с правильным viewBox
    drawing = svgwrite.Drawing(filename, (width, height), debug=True)
    drawing.viewbox(x_min - margin_left, y_min - margin_top,
                    bb2d_width + margin_left + margin_right,
                    bb2d_height + margin_top + margin_bottom)
    
    # Добавляем геометрию
    for path in paths:
        path.stroke(color, width=line_width, linecap="round")
        drawing.add(path)
    
    # Добавляем размеры
    if add_tolerances:
        if tolerance_manager is None:
            tolerance_manager = ToleranceManager(seed=42)
        
        # Масштаб текста относительно размера чертежа
        text_scale = max(0.5, min(2.0, bb2d_width / 200))
        font_size = 10 * text_scale
        arrow_size = 8 * text_scale
        dim_offset = 20 * text_scale
        
        # Горизонтальный размер (ширина) - внизу
        width_value = round(bb2d_width, 2)
        width_tolerance = tolerance_manager.get_tolerance(width_value)
        
        dim_line_y = y_max + dim_offset
        
        ext_line1 = svgwrite.shapes.Line((x_min, y_max), (x_min, dim_line_y + 5 * text_scale), stroke='black', stroke_width=0.35)
        ext_line2 = svgwrite.shapes.Line((x_max, y_max), (x_max, dim_line_y + 5 * text_scale), stroke='black', stroke_width=0.35)
        drawing.add(ext_line1)
        drawing.add(ext_line2)
        
        drawing.add(svgwrite.shapes.Line((x_min, dim_line_y), (x_max, dim_line_y), stroke='black', stroke_width=0.35))
        
        left_arrow = svgwrite.path.Path(d=f"M {x_min},{dim_line_y} L {x_min + arrow_size},{dim_line_y - arrow_size * 0.4} L {x_min + arrow_size},{dim_line_y + arrow_size * 0.4} Z", fill='black')
        right_arrow = svgwrite.path.Path(d=f"M {x_max},{dim_line_y} L {x_max - arrow_size},{dim_line_y - arrow_size * 0.4} L {x_max - arrow_size},{dim_line_y + arrow_size * 0.4} Z", fill='black')
        drawing.add(left_arrow)
        drawing.add(right_arrow)
        
        dim_text_str = f"{width_value:.2f}"
        if width_tolerance.get('type') == 'bilateral':
            dim_text_str += f"±{width_tolerance['value']:.2f}"
        elif width_tolerance.get('type') == 'unilateral_upper':
            dim_text_str += f"+{width_tolerance['upper_limit'] - width_value:.2f}/0"
        elif width_tolerance.get('type') == 'unilateral_lower':
            dim_text_str += f"0/-{width_value - width_tolerance['lower_limit']:.2f}"
            
        drawing.add(svgwrite.shapes.Rect(insert=(center_x - 25 * text_scale, dim_line_y - font_size * 0.8), size=(50 * text_scale, font_size * 1.6), fill='white', stroke='none'))
        drawing.add(svgwrite.text.Text(dim_text_str, insert=(center_x, dim_line_y + font_size * 0.3), font_size=str(font_size), fill='black', text_anchor='middle', font_family='Arial, sans-serif'))
        
        # Вертикальный размер (высота) - слева
        height_value = round(bb2d_height, 2)
        height_tolerance = tolerance_manager.get_tolerance(height_value)
        v_dim_line_x = x_min - dim_offset
        
        drawing.add(svgwrite.shapes.Line((v_dim_line_x, y_min), (v_dim_line_x, y_max), stroke='black', stroke_width=0.35))
        drawing.add(svgwrite.shapes.Line((x_min, y_min), (v_dim_line_x - 5 * text_scale, y_min), stroke='black', stroke_width=0.35))
        drawing.add(svgwrite.shapes.Line((x_min, y_max), (v_dim_line_x - 5 * text_scale, y_max), stroke='black', stroke_width=0.35))
        
        top_arrow = svgwrite.path.Path(d=f"M {v_dim_line_x},{y_min} L {v_dim_line_x - arrow_size * 0.4},{y_min + arrow_size} L {v_dim_line_x + arrow_size * 0.4},{y_min + arrow_size} Z", fill='black')
        bottom_arrow = svgwrite.path.Path(d=f"M {v_dim_line_x},{y_max} L {v_dim_line_x - arrow_size * 0.4},{y_max - arrow_size} L {v_dim_line_x + arrow_size * 0.4},{y_max - arrow_size} Z", fill='black')
        drawing.add(top_arrow)
        drawing.add(bottom_arrow)
        
        v_dim_text_str = f"{height_value:.2f}"
        if height_tolerance.get('type') == 'bilateral':
            v_dim_text_str += f"±{height_tolerance['value']:.2f}"
        elif height_tolerance.get('type') == 'unilateral_upper':
            v_dim_text_str += f"+{height_tolerance['upper_limit'] - height_value:.2f}/0"
        elif height_tolerance.get('type') == 'unilateral_lower':
            v_dim_text_str += f"0/-{height_value - height_tolerance['lower_limit']:.2f}"
            
        drawing.add(svgwrite.shapes.Rect(insert=(v_dim_line_x - font_size * 1.5, center_y - font_size * 0.8), size=(font_size * 3, font_size * 1.6), fill='white', stroke='none'))
        drawing.add(svgwrite.text.Text(v_dim_text_str, insert=(v_dim_line_x - font_size * 0.5, center_y), font_size=str(font_size), fill='black', text_anchor='middle', font_family='Arial, sans-serif', transform=f'rotate(-90, {v_dim_line_x - font_size * 0.5}, {center_y})'))
    
    if filename is not None:
        drawing.save()
        return filename
    return drawing.tostring()

# ==================== Вспомогательные функции ====================
def read_step_file(step_file: str):
    """Чтение STEP файла."""
    reader = STEPControl_Reader()
    status = reader.ReadFile(step_file)
    
    if status != IFSelect_RetDone:
        raise ValueError(f"Cannot read STEP file: {step_file}")
    
    reader.TransferRoots()
    shape = reader.OneShape()
    
    return shape


# ==================== Основная функция для генерации всех видов ====================
def generate_all_views(step_file: str, output_dir: str, seed: int = 42) -> Dict:
    """
    Генерирует все виды (front, top, side) с консистентными допусками.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Создаем ОДИН менеджер допусков для всех видов
    tolerance_manager = ToleranceManager(seed=seed)
    
    # Читаем STEP-файл
    shape = read_step_file(step_file)
    
    base_name = os.path.splitext(os.path.basename(step_file))[0]
    
    views = {
        'front': (0, 0, 1),
        'top': (0, 1, 0),
        'side': (1, 0, 0)
    }
    
    generated_files = {}
    
    for view_name, view_direction in views.items():
        svg_file = os.path.join(output_dir, f"{base_name}_{view_name}.svg")
        
        try:
            export_shape_to_svg_with_tolerances(
                shape,
                svg_file,
                direction=view_direction,
                width=800,
                height=600,
                tolerance_manager=tolerance_manager
            )
            generated_files[view_name] = svg_file
            print(f"✓ Generated {view_name} view: {svg_file}")
        except Exception as e:
            print(f"  Failed to generate {view_name} view: {e}")
            generated_files[view_name] = None
    
    # Сохраняем кэш допусков для верификации
    tolerance_cache_file = os.path.join(output_dir, f"{base_name}_tolerance_cache.json")
    
    # Конвертируем numpy-типы перед сохранением
    def convert_numpy(obj):
        if isinstance(obj, dict):
            return {convert_numpy(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj
    
    cache_to_save = convert_numpy(tolerance_manager.tolerance_cache)
    with open(tolerance_cache_file, 'w') as f:
        json.dump(cache_to_save, f, indent=2)
    print(f"✓ Saved tolerance cache: {tolerance_cache_file}")
    
    return generated_files


# ==================== Тестирование ====================
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate engineering drawings from STEP files')
    parser.add_argument('-i', '--input', required=True, help='Input STEP file')
    parser.add_argument('-o', '--output', required=True, help='Output directory')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    generate_all_views(args.input, args.output, args.seed)