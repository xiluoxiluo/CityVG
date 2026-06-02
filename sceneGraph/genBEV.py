import torch
import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm

block_name = "birmingham_block_9"
data_root = "/home/zjj/Code/CityVG/data/sensaturban"
orig_path = os.path.join(data_root, f"{block_name}.ply") 
pth_path = "/home/zjj/Code/CityAnchor/data/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250/birmingham_block_9.pth"

save_root = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_preprocessed", block_name)
os.makedirs(save_root, exist_ok=True)

# === 参数设置 ===
resolution = 0.1
draw_thickness = 3
expand_list = [20.0]     # 只生成 expand = 5
bbox_padding_px = 1     # 红框留间隙

# ←←← 新增：控制是否绘制红框
DRAW_RED_BOX = False

# === 加载点云数据 ===
print("📥 读取点云与 bbox 数据...")
pcd_orig = o3d.io.read_point_cloud(orig_path)
pts_orig = np.asarray(pcd_orig.points)
cols_orig = np.asarray(pcd_orig.colors)

# === 读取 .pth 文件 ===
coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, \
instance_bboxes, landmark_names, landmark_ids, globalShift = torch.load(
    pth_path, map_location="cpu", weights_only=False
)

bbox_items = list(instance_bboxes.items())
subset = bbox_items
print(f"📦 总实例数: {len(instance_bboxes)}，本次处理数量: {len(subset)}")


# ============================================================
#                 主循环：生成 canvas.jpg
# ============================================================

for inst_id, bbox in tqdm(subset[0:1], desc="Generating topviews", ncols=100):

    cx, cy, cz, sx, sy, sz, label_id, _ = bbox
    bbox[:3] += globalShift

    cx, cy, cz, sx, sy, sz = bbox[:6]
    corner1 = np.array([cx - sx / 2, cy - sy / 2])
    corner2 = np.array([cx + sx / 2, cy + sy / 2])

    inst_dir = os.path.join(
        "/home/zjj/Code/CityVG/data/cityrefer_preprocessed/birmingham_block_9",
        str(int(inst_id))
    )
    os.makedirs(inst_dir, exist_ok=True)

    for expand_padding in expand_list:

        x_min, x_max = sorted([corner1[0], corner2[0]])
        y_min, y_max = sorted([corner1[1], corner2[1]])

        x_min -= expand_padding
        x_max += expand_padding
        y_min -= expand_padding
        y_max += expand_padding

        mask = (
            (pts_orig[:, 0] >= x_min) & (pts_orig[:, 0] <= x_max) &
            (pts_orig[:, 1] >= y_min) & (pts_orig[:, 1] <= y_max)
        )
        pts_crop = pts_orig[mask]
        cols_crop = cols_orig[mask]

        if len(pts_crop) == 0:
            continue

        width_px = int((x_max - x_min) / resolution)
        height_px = int((y_max - y_min) / resolution)
        raster = np.zeros((height_px, width_px, 3), dtype=np.uint8)

        # ----- 投影点云 -----
        for p, c in zip(pts_crop, cols_crop):
            ix = int((p[0] - x_min) / resolution)
            iy = int((y_max - p[1]) / resolution)
            if 0 <= ix < width_px and 0 <= iy < height_px:
                raster[iy, ix, :] = (c * 255).astype(np.uint8)

        # =====================================================
        #             绘制红框（可关闭）
        # =====================================================
        img_bbox = cv2.cvtColor(raster.copy(), cv2.COLOR_RGB2BGR)

        if DRAW_RED_BOX:
            ix_min = int((corner1[0] - x_min) / resolution) - bbox_padding_px
            ix_max = int((corner2[0] - x_min) / resolution) + bbox_padding_px
            iy_min = int((y_max - corner2[1]) / resolution) - bbox_padding_px
            iy_max = int((y_max - corner1[1]) / resolution) + bbox_padding_px

            ix_min = max(ix_min, 0)
            iy_min = max(iy_min, 0)
            ix_max = min(ix_max, width_px - 1)
            iy_max = min(iy_max, height_px - 1)

            cv2.rectangle(
                img_bbox,
                (ix_min, iy_min),
                (ix_max, iy_max),
                (0, 0, 255),
                draw_thickness
            )

        # ---- 保存 ----
        img_path = os.path.join(inst_dir, f"canvas_raw_{int(expand_padding)}.jpg")
        cv2.imwrite(img_path, img_bbox)

print(f"\n🎉 指定实例生成完成，输出目录：{save_root}")
