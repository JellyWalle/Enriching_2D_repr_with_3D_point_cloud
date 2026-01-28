import glob
from tqdm import tqdm  # ← правильно!
import os
from icp import evaluate_icp_on_scene
import numpy as np

OUTPUT_DIR = '/mnt/hdd_works/ml_640/yolov11ir/abc_full_10k_v00/10k/dataset_generated_final_pyrender'
ann_files = glob.glob(os.path.join(OUTPUT_DIR, 'annotations', '*.json'))
errors = []

for ann_file in tqdm(ann_files):
    model_name = os.path.splitext(os.path.basename(ann_file))[0]
    pc_file = os.path.join(OUTPUT_DIR, 'points', f"{model_name}.npy")
    if not os.path.exists(pc_file):
        continue
    try:
        err = evaluate_icp_on_scene(ann_file, pc_file)
        errors.append(err)
    except Exception as e:
        print(f"Skip {model_name}: {e}")

mean_error = np.mean(errors)
print(f"Mean angular error (ICP + GT): {mean_error:.2f}°")