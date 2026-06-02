import torch
import open3d as o3d
import numpy as np
import os
import cv2
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--SCENE_ID", type=str, required=True)
args = parser.parse_args()
SCENE_ID = args.SCENE_ID
data_root = "/home/zjj/Code/CityVG/data/sensaturban"
orig_path = os.path.join(data_root, f"{SCENE_ID}.ply") 
sem_path = os.path.join(data_root, f"{SCENE_ID}_semantic.ply")
pth_path = os.path.join("/home/zjj/Code/CityAnchor/data/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250", f"{SCENE_ID}.pth")

save_root = os.path.join("/home/zjj/Code/CityVG/data/cityrefer_preprocessed", SCENE_ID)
os.makedirs(save_root, exist_ok=True)

resolution = 0.1
draw_thickness = 3
#expand_list = [5, 20, 40]  
expand_list = [100] 

pcd_orig = o3d.io.read_point_cloud(orig_path)
pcd_sem = o3d.io.read_point_cloud(sem_path)
pts_orig = np.asarray(pcd_orig.points)
cols_orig = np.asarray(pcd_orig.colors)
pts_sem = np.asarray(pcd_sem.points)
cols_sem = np.asarray(pcd_sem.colors)

coords, colors, label_ids, instance_ids, label_ids_pg, instance_ids_pg, instance_bboxes, landmark_names, landmark_ids, globalShift = torch.load(
    pth_path, map_location="cpu", weights_only=False
)
print(f"📦 Total Instances: {len(instance_bboxes)}")

for inst_id, bbox in tqdm(instance_bboxes.items(), desc="Generating topviews", ncols=100):

    cx, cy, cz, sx, sy, sz, label_id, _ = bbox
    bbox[:3] += globalShift  
    cx, cy, cz, sx, sy, sz = bbox[:6]
    corner1 = np.array([cx - sx / 2, cy - sy / 2])
    corner2 = np.array([cx + sx / 2, cy + sy / 2])

    inst_dir = os.path.join(save_root, str(int(inst_id)))
    os.makedirs(inst_dir, exist_ok=True)

    for expand_padding in tqdm(expand_list, desc=f"Instance {int(inst_id)}", leave=False, ncols=90):

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

        for p, c in zip(pts_crop, cols_crop):
            ix = int((p[0] - x_min) / resolution)
            iy = int((y_max - p[1]) / resolution)
            if 0 <= ix < width_px and 0 <= iy < height_px:
                raster[iy, ix, :] = (c * 255).astype(np.uint8)

        raster_rgb = cv2.cvtColor(raster.copy(), cv2.COLOR_RGB2BGR)

        ix_min = int((corner1[0] - x_min) / resolution)
        ix_max = int((corner2[0] - x_min) / resolution)
        iy_min = int((y_max - corner2[1]) / resolution)
        iy_max = int((y_max - corner1[1]) / resolution)
        ix_min, ix_max = sorted([ix_min, ix_max])
        iy_min, iy_max = sorted([iy_min, iy_max])

        if expand_padding == expand_list[0]:
            raw_path = os.path.join(inst_dir, f"canvas_raw_{int(expand_padding)}.jpg")
            cv2.imwrite(raw_path, raster_rgb)

        img_bbox = raster_rgb.copy()
        cv2.rectangle(img_bbox, (ix_min, iy_min), (ix_max, iy_max), (0, 0, 255), draw_thickness)

        img_path = os.path.join(inst_dir, f"canvas_{int(expand_padding)}.jpg")
        cv2.imwrite(img_path, img_bbox)

        if expand_padding == 40:
            mask_sem = (
                (pts_sem[:, 0] >= x_min) & (pts_sem[:, 0] <= x_max) &
                (pts_sem[:, 1] >= y_min) & (pts_sem[:, 1] <= y_max)
            )
            pts_sem_crop = pts_sem[mask_sem]
            cols_sem_crop = cols_sem[mask_sem]

            raster_sem = np.zeros((height_px, width_px, 3), dtype=np.uint8)
            for p, c in zip(pts_sem_crop, cols_sem_crop):
                ix = int((p[0] - x_min) / resolution)
                iy = int((y_max - p[1]) / resolution)
                if 0 <= ix < width_px and 0 <= iy < height_px:
                    raster_sem[iy, ix, :] = (c * 255).astype(np.uint8)

            sem_rgb = cv2.cvtColor(raster_sem.copy(), cv2.COLOR_RGB2BGR)
            cv2.rectangle(sem_rgb, (ix_min, iy_min), (ix_max, iy_max), (0, 0, 255), draw_thickness)

            sem_path_out = os.path.join(inst_dir, "semantic_topview.jpg")
            cv2.imwrite(sem_path_out, sem_rgb)

print(f"\n💾 Saved → {save_root}")

print("****************************************")
print("STEP 3/9: Extract instances and generate center-aligned top-view projections")
print("****************************************")
