# Hybrid GNN Dataset Generator

Полный конвейер обработки данных для обучения гибридной графовой нейронной сети (GNN) 
для верификации соответствия 2D чертежей и 3D облаков точек.

## Структура проекта

```
itmo/
├── data/
│   ├── abc_dataset/              # Исходные STEP файлы
│   ├── drawings/                 # Сгенерированные 2D чертежи
│   ├── pointclouds/              # Сгенерированные облака точек
│   ├── annotations/              # Разметка соответствий
│   ├── graphs/                   # Гибридные графы
│   └── processed/                # Обработанные данные
│       ├── features_2d/          # 2D признаки
│       ├── features_3d/          # 3D признаки
│       ├── train/                # Обучающий набор (70%)
│       ├── val/                  # Валидационный набор (20%)
│       └── test/                 # Тестовый набор (10%)
├── src/
│   ├── data_generation/
│   │   └── generate_data.py      # Генерация 2D чертежей и 3D облаков
│   ├── feature_extraction/
│   │   ├── feature_2d.py         # Извлечение 2D признаков (YOLOv7 + OCR)
│   │   ├── feature_3d.py         # Извлечение 3D признаков (PointNet++)
│   │   └── graph_builder.py      # Построение гибридного графа
│   ├── preprocessing/            # Предварительная обработка
│   ├── models/                   # GNN модели
│   └── utils/                    # Вспомогательные функции
├── scripts/
│   └── run_pipeline.py           # Главный скрипт конвейера
├── utils/                        # Существующие утилиты
├── requirements.txt              # Зависимости
└── README.md                     # Этот файл
```

## Установка

```bash
pip install -r requirements.txt
```

## Быстрый старт

### 1. Генерация полного датасета

```bash
python scripts/run_pipeline.py \
    -i /path/to/abc_dataset \
    -o /path/to/output \
    --max-files 150 \
    --num-points 1024
```

### 2. Отдельные этапы

#### Генерация данных
```bash
python src/data_generation/generate_data.py \
    -i /path/to/abc_dataset \
    -o /path/to/output \
    --max-files 150
```

#### Извлечение 2D признаков
```bash
python src/feature_extraction/feature_2d.py \
    -i data/drawings/model_001_front.svg \
    -o data/processed/features_2d/model_001_front_features.json
```

#### Извлечение 3D признаков
```bash
python src/feature_extraction/feature_3d.py \
    -i data/pointclouds/model_001_pc_ideal.npy \
    -o data/processed/features_3d/model_001_ideal_features.json
```

#### Построение графа
```bash
python src/feature_extraction/graph_builder.py \
    --features-2d data/processed/features_2d/model_001_front_features.json \
    --features-3d data/processed/features_3d/model_001_ideal_features.json \
    -o data/graphs/model_001_front_graph.json
```

## Этапы конвейера

### Этап 1: Генерация данных
- **Вход**: STEP файлы из ABC dataset
- **Выход**: 
  - 2D чертежи (SVG) с видами спереди/сверху/сбоку
  - 3D облака точек (NPY) с идеальными/допустимыми/недопустимыми отклонениями
  - Аннотации (JSON)

### Этап 2: Извлечение 2D признаков
- **Вход**: SVG чертежи
- **Выход**: Node2D объекты с размерами, допусками, позициями
- **Методы**: 
  - Image preprocessing (grayscale, binarization, denoising)
  - YOLOv7 detection (dimensions, tolerances)
  - Tesseract OCR (text extraction)

### Этап 3: Извлечение 3D признаков
- **Вход**: NPY облака точек
- **Выход**: Node3D объекты с производственными признаками
- **Методы**:
  - Point cloud preprocessing (outlier removal, normalization, downsampling)
  - Geometric segmentation (normals, curvature, PCA)
  - Feature classification (hole, slot, pocket, etc.)

### Этап 4: Построение гибридного графа
- **Вход**: 2D и 3D признаки
- **Выход**: HybridGraph с correspondence edges
- **Методы**:
  - Spatial matching (projection + distance)
  - Semantic matching (type similarity)
  - Geometric matching (dimension similarity)

### Этап 5: Разделение на наборы
- **Train**: 70% (105 деталей × 3 вида = 315 графов)
- **Val**: 20% (30 деталей × 3 вида = 90 графов)
- **Test**: 10% (15 деталей × 3 вида = 45 графов)

## Структуры данных

### Node2D (2D признак)
```json
{
  "id": "node2d_001",
  "type": "dimension",
  "value": 10.0,
  "tolerance": {"type": "bilateral", "value": 0.1},
  "position_2d": [150, 200],
  "feature_type": "hole",
  "bounding_box": [140, 190, 160, 210],
  "text": "⌀10±0.1",
  "confidence": 0.95
}
```

### Node3D (3D признак)
```json
{
  "id": "node3d_001",
  "type": "geometry",
  "feature_type": "hole",
  "measured_value": 10.05,
  "position_3d": [0.0, 0.0, 5.0],
  "normals": [0.0, 0.0, 1.0],
  "curvature": 0.15,
  "num_points": 128,
  "point_indices": [0, 1, 2, ...],
  "confidence": 0.92
}
```

### EdgeCorrespondence (ребро соответствия)
```json
{
  "source_id": "node2d_001",
  "target_id": "node3d_001",
  "weight": 0.85,
  "deviation": 0.05,
  "spatial_distance": 0.12,
  "semantic_similarity": 0.95,
  "type": "correspondence"
}
```

### HybridGraph
```json
{
  "num_2d_nodes": 15,
  "num_3d_nodes": 12,
  "num_edges": 10,
  "nodes_2d": {...},
  "nodes_3d": {...},
  "edges": [...]
}
```

## Зависимости

### Основные
- numpy >= 1.20
- scipy >= 1.7
- opencv-python >= 4.5
- tqdm >= 4.62

### Для 2D обработки
- ultralytics (YOLOv7)
- pytesseract (OCR)

### Для 3D обработки
- pythonocc-core (STEP files)
- open3d (point clouds)

### Для GNN
- torch >= 1.9
- torch-geometric >= 2.0

## Лицензия

MIT License
