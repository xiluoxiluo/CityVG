# =============================
#   四方向航拍倾斜摄影（正交）
# =============================

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

resolution = 0.1
expand_list = [20.0]

# 航拍倾斜角（越大越倾斜）
tilt_angle_deg = 45

# 相机朝向
view_dirs = {
    "north": np.array([0, 1, 0]),
    "south": np.array([0,-1, 0]),
    "east":  np.array([1, 0, 0]),
    "west":  np.array([-1,0, 0])
}

z_up = np.array([0,0,1])      # 世界竖直方向

# =============== 加载数据 ===============
pcd_orig = o3d.io.read_point_cloud(orig_path)
pts_orig = np.asarray(pcd_orig.points)
cols_orig = np.asarray(pcd_orig.colors)

(coords, colors, label_ids, instance_ids,
 label_ids_pg, instance_ids_pg, instance_bboxes,
 landmark_names, landmark_ids, globalShift) = torch.load(
     pth_path, map_location="cpu", weights_only=False
 )

bbox_items = list(instance_bboxes.items())


# ============================================================
#            主循环：生成航拍倾斜摄影图像
# ============================================================

for inst_id, bbox in tqdm(bbox_items):

    cx, cy, cz, sx, sy, sz, label_id, _ = bbox
    bbox[:3] += globalShift
    cx, cy, cz, sx, sy, sz = bbox[:6]
    center = np.array([cx, cy, cz])

    corner1 = np.array([cx - sx/2, cy - sy/2])
    corner2 = np.array([cx + sx/2, cy + sy/2])

    inst_dir = os.path.join(save_root, str(int(inst_id)))
    os.makedirs(inst_dir, exist_ok=True)

    for expand_padding in expand_list:

        # crop bounding box
        x_min, x_max = sorted([corner1[0], corner2[0]])
        y_min, y_max = sorted([corner1[1], corner2[1]])

        x_min -= expand_padding
        x_max += expand_padding
        y_min -= expand_padding
        y_max += expand_padding

        mask = (
            (pts_orig[:,0] >= x_min) & (pts_orig[:,0] <= x_max) &
            (pts_orig[:,1] >= y_min) & (pts_orig[:,1] <= y_max)
        )

        pts_crop = pts_orig[mask]
        cols_crop = cols_orig[mask]

        pts_local = pts_crop - center

        # =====================================================
        #      为该实例生成四方向航拍倾斜图
        # =====================================================
        for view_name, ground_dir in view_dirs.items():

            # -------- 构造相机 forward 向量（倾斜 a°） --------
            phi = np.radians(tilt_angle_deg)

            # 原本 forward = -ground_dir
            forward0 = -ground_dir
            forward = np.cos(phi)*forward0 + np.sin(phi)*(-z_up)
            forward = forward / np.linalg.norm(forward)

            # -------- 相机 right/up --------
            right = np.cross(forward, z_up)
            right = right / np.linalg.norm(right)

            up_cam = np.cross(right, forward)

            # -------- 投影到相机坐标 --------
            u = pts_local @ right
            v = pts_local @ up_cam

            # -------- 转成像素坐标 --------
            pad = 1.0
            u_min, u_max = u.min()-pad, u.max()+pad
            v_min, v_max = v.min()-pad, v.max()+pad

            width_px  = int((u_max-u_min)/resolution)
            height_px = int((v_max-v_min)/resolution)

            img = np.zeros((height_px, width_px, 3), dtype=np.uint8)

            for uu, vv, c in zip(u, v, cols_crop):
                ix = int((uu - u_min) / resolution)
                iy = int((v_max - vv) / resolution)

                if 0<=ix<width_px and 0<=iy<height_px:
                    img[iy, ix, :] = (c*255).astype(np.uint8)

            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            outpath = os.path.join(inst_dir, f"{view_name}_tilt_{tilt_angle_deg}deg.jpg")
            cv2.imwrite(outpath, img_bgr)

print("\n🎉 航拍倾斜视角图像全部生成！")
