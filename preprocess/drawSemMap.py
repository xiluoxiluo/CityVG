import numpy as np
import os
from helper_ply import read_ply, write_ply
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--SCENE_ID", type=str, required=True)
args = parser.parse_args()
SCENE_ID = args.SCENE_ID

input_ply = f"/home/zjj/Code/CityVG/data/sensaturban/{SCENE_ID}.ply" 
out_ply = f"/home/zjj/Code/CityVG/data/sensaturban/{SCENE_ID}_semantic.ply"
data = read_ply(input_ply)

# === 2. 提取字段 ===
x = data['x']
y = data['y']
z = data['z']
cls = data['class']

semantic_colors = np.array([
    [85, 107, 47],    # ground -> OliveDrab
    [0, 255, 0],      # tree -> Green
    [255, 165, 0],    # building -> orange
    [41, 49, 101],    # Walls -> darkblue
    [0, 0, 0],        # Bridge -> black
    [0, 0, 255],      # parking -> blue
    [255, 0, 255],    # rail -> Magenta
    [200, 200, 200],  # traffic Roads -> grey
    [89, 47, 95],     # Street Furniture -> DimGray
    [255, 0, 0],      # cars -> red
    [255, 255, 0],    # Footpath -> yellow
    [0, 255, 255],    # bikes -> cyan
    [0, 191, 255]     # water -> skyblue
], dtype=np.uint8)

cls_valid = np.clip(cls, 0, len(semantic_colors)-1)
rgb = semantic_colors[cls_valid]

write_ply(out_ply, [np.vstack((x, y, z)).T, rgb], ['x', 'y', 'z', 'red', 'green', 'blue'])
print(f"\n💾 Saved → {out_ply}")

print("****************************************")
print("STEP 2/9: Generate semantic visualization for the scene")
print("****************************************")
