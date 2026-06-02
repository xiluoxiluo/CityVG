from glob import glob
import os
import cv2
from tqdm import tqdm
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--SCENE_ID", type=str, required=True)
args = parser.parse_args()
SCENE_ID = args.SCENE_ID

save_root = f"/home/zjj/Code/CityVG/data/cityrefer_preprocessed/{SCENE_ID}"
# === 间隔与目标尺寸设置 ===
gap_height = 10               # 每张之间的空白间隔（像素）
gap_color = (255, 255, 255)    # 间隔颜色（白色）
target_width = 1296
target_height = 5369

# === 遍历每个实例文件夹 ===
inst_dirs = sorted([d for d in os.listdir(save_root) if os.path.isdir(os.path.join(save_root, d))])

for inst in tqdm(inst_dirs, desc="Merging canvases", ncols=90):
    inst_dir = os.path.join(save_root, inst)
    
    # 要拼接的图片路径（按顺序）
    img_names = ["canvas_raw_5.jpg","canvas_5.jpg", "canvas_20.jpg", "canvas_40.jpg", "semantic_topview.jpg"]
    img_paths = [os.path.join(inst_dir, name) for name in img_names if os.path.exists(os.path.join(inst_dir, name))]

    if len(img_paths) == 0:
        continue

    # === 读取图像 ===
    imgs = [cv2.imread(p) for p in img_paths if cv2.imread(p) is not None]

    # === 统一宽度为 target_width ===
    imgs_resized = [cv2.resize(img, (target_width, int(img.shape[0] * target_width / img.shape[1]))) for img in imgs]

    # === 构造间隔条 ===
    gap = np.full((gap_height, target_width, 3), gap_color, dtype=np.uint8)

    # === 拼接（插入空白间隔） ===
    merged = []
    for i, img in enumerate(imgs_resized):
        merged.append(img)
        if i != len(imgs_resized) - 1:
            merged.append(gap)
    canvas_all = cv2.vconcat(merged)

    canvas_final = cv2.resize(canvas_all, (target_width, target_height), interpolation=cv2.INTER_AREA)

    out_path = os.path.join(inst_dir, "canvas.jpg")
    cv2.imwrite(out_path, canvas_final)

print(f"✅Concat Finshed: {save_root}")

print("****************************************")
print("STEP 4/9: Construct multi-scale top-view representation")
print("****************************************")
