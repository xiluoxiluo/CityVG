#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Auto Landmark BEV Generator (Purple Box Version)
-------------------------------------------------
功能：
1. 自动解析 birmingham_block_4_bbox.json 得到 id_map 与 landmark_list
2. 自动调用 BEV 生成流程，为每个 landmark 生成紫色框 BEV
3. 输出结构：
   data/cityrefer_preprocessed/<block>/<object_id>/landmark_bev.jpg
"""

import os
import json
import torch
import open3d as o3d
import numpy as np
import cv2


# ============================================================
# 📌 配置（按你的项目路径保持一致）
# ============================================================

block_name = "birmingham_block_4"

json_path = f"/home/zjj/Code/CityVG/data/{block_name}_bbox.json"

owndata_root = "/home/zjj/Code/CityAnchor/data/sensaturban"
ply_path = os.path.join(owndata_root, f"{block_name}.ply")

pth_path = "/home/zjj/Code/CityAnchor/data/data_cityrefer/sensaturban/" \
           "pointgroup_data/balance_split/random-50_crop-250/birmingham_block_4.pth"

save_root = f"/home/zjj/Code/CityVG/data/cityrefer_preprocessed/{block_name}"
os.makedirs(save_root, exist_ok=True)


# ============================================================
# Step 1 — 自动生成 id_map 与 landmark_list
# ============================================================

def extract_id_map(json_path):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # 必须进入 data["bboxes"]
    if "bboxes" not in data:
        raise ValueError("JSON 文件格式不符合预期，未找到 'bboxes' 字段")

    id_map = {}
    landmark_list = []

    for item in data["bboxes"]:
        lm = item.get("landmark", "")
        if not isinstance(lm, str):  # 如果是 None 或其它类型
            lm = str(lm)

        lm = lm.strip()
        obj_id = item.get("object_id")

        if lm != "":
            id_map[lm] = obj_id
            landmark_list.append(lm)
    print(id_map)
    return id_map, landmark_list

print("📥 Extracting id_map and landmark_list ...")
id_map, landmark_list = extract_id_map(json_path)

print(f"🎉 Found {len(id_map)} landmarks:")
for k, v in id_map.items():
    print(f"  - {k} → object_id={v}")


# ============================================================
# Step 2 — BEV 生成参数
# ============================================================

resolution = 0.1
expand_padding = 8.0
draw_thickness = 3
BOX_COLOR = (255, 0, 255)   # Purple box
bbox_padding_px = 2
global_shift = np.array([400.0, 400.0, 5.17])


# ============================================================
# Step 3 — 加载点云与 PTH
# ============================================================

print("\n📥 Loading point cloud ...")
pcd_orig = o3d.io.read_point_cloud(ply_path)
pts_orig = np.asarray(pcd_orig.points)
cols_orig = np.asarray(pcd_orig.colors)

print("📥 Loading instance bbox data (.pth) ...")
(
    coords, colors, label_ids, instance_ids,
    label_ids_pg, instance_ids_pg,
    instance_bboxes, landmark_names, landmark_ids, globalShift
) = torch.load(pth_path, map_location="cpu", weights_only=False)

# 转 list
instance_bboxes = list(instance_bboxes.items())


# ============================================================
# Step 4 — 主函数：生成单个 landmark 的紫色框 BEV
# ============================================================

def generate_landmark_bev(landmark_name, landmark_obj_id):
    print(f"\n🎯 Generating BEV for landmark: '{landmark_name}' (id={landmark_obj_id})")

    # 找 bbox
    target_bbox = None
    for inst_id, bbox in instance_bboxes:
        if int(inst_id) == int(landmark_obj_id):
            target_bbox = bbox
            break

    if target_bbox is None:
        print(f"⚠️ No bbox found for object_id={landmark_obj_id}, skipped.")
        return None

    # shift
    bbox = target_bbox.copy()
    bbox[:3] += global_shift

    cx, cy, cz, sx, sy, sz = bbox[:6]
    corner1 = np.array([cx - sx / 2, cy - sy / 2])
    corner2 = np.array([cx + sx / 2, cy + sy / 2])

    # padding
    x_min, x_max = sorted([corner1[0], corner2[0]])
    y_min, y_max = sorted([corner1[1], corner2[1]])
    x_min -= expand_padding
    x_max += expand_padding
    y_min -= expand_padding
    y_max += expand_padding

    # crop
    mask = (
        (pts_orig[:, 0] >= x_min) & (pts_orig[:, 0] <= x_max) &
        (pts_orig[:, 1] >= y_min) & (pts_orig[:, 1] <= y_max)
    )
    pts_crop = pts_orig[mask]
    cols_crop = cols_orig[mask]

    if len(pts_crop) == 0:
        print("⚠️ BEV region empty, skipped.")
        return None

    # raster size
    width_px = int((x_max - x_min) / resolution)
    height_px = int((y_max - y_min) / resolution)
    raster = np.zeros((height_px, width_px, 3), dtype=np.uint8)

    # projection
    for p, c in zip(pts_crop, cols_crop):
        ix = int((p[0] - x_min) / resolution)
        iy = int((y_max - p[1]) / resolution)
        if 0 <= ix < width_px and 0 <= iy < height_px:
            raster[iy, ix] = (c * 255).astype(np.uint8)

    img = cv2.cvtColor(raster, cv2.COLOR_RGB2BGR)

    # draw purple box
    ix_min = int((corner1[0] - x_min) / resolution) - bbox_padding_px
    ix_max = int((corner2[0] - x_min) / resolution) + bbox_padding_px
    iy_min = int((y_max - corner2[1]) / resolution) - bbox_padding_px
    iy_max = int((y_max - corner1[1]) / resolution) + bbox_padding_px

    cv2.rectangle(img, (ix_min, iy_min), (ix_max, iy_max), BOX_COLOR, draw_thickness)

    # save
    out_dir = os.path.join(save_root, str(int(landmark_obj_id)))
    os.makedirs(out_dir, exist_ok=True)
    save_path = os.path.join(out_dir, "landmark_bev.jpg")
    cv2.imwrite(save_path, img)

    print(f"✅ Saved: {save_path}")
    return save_path


# ============================================================
# Step 5 — 对所有 landmark 批量生成 BEV
# ============================================================

print("\n🚀 Generating BEV for all landmarks ...")

for lm, obj_id in id_map.items():
    generate_landmark_bev(lm, obj_id)

print("\n🎉 All landmark BEVs generated successfully!")

