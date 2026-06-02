import json
import torch
from collections import defaultdict

# =========================
# 1. 路径设置
# =========================
block_name = "birmingham_block_9"

pth_path = f"/home/zjj/Code/CityAnchor/data/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250/{block_name}.pth"
json_path = f"/home/zjj/Code/CityVG/data/cityrefer_inference/{block_name}_inference.json"

# =========================
# 2. 加载 pth
# =========================
(
    coords, colors, label_ids, instance_ids,
    label_ids_pg, instance_ids_pg,
    instance_bboxes, landmark_names,
    landmark_ids, globalShift
) = torch.load(pth_path, map_location="cpu", weights_only=False)

# =========================
# 3. Category 归一化函数
# =========================
def normalize_category(name: str) -> str:
    name = name.lower()

    if "car" in name:
        return "Car"
    if "building" in name:
        return "Building"
    if "parking" in name:
        return "Parking"
    if "ground" in name or "road" in name or "terrain" in name:
        return "Ground"

    return "Other"

# =========================
# 4. 从 landmark_names 构建 ID → Category
# =========================
ID2CATEGORY = {}

if isinstance(landmark_names, dict):
    # dict: {id: name}
    for idx, name in landmark_names.items():
        ID2CATEGORY[int(idx)] = normalize_category(name)

elif isinstance(landmark_names, list):
    # list: index = id
    for idx, name in enumerate(landmark_names):
        ID2CATEGORY[idx] = normalize_category(name)

else:
    raise TypeError(f"Unsupported landmark_names type: {type(landmark_names)}")

# =========================
# 5. 读取 JSON 推理结果
# =========================
with open(json_path, "r") as f:
    results = json.load(f)

# =========================
# 6. 统计 Overall / Per-class ACC
# =========================
total = 0
correct = 0

class_total = defaultdict(int)
class_correct = defaultdict(int)

for item in results:
    if "GTID" not in item or "best_id" not in item:
        continue

    gt = int(item["GTID"])
    pred = int(item["best_id"])

    category = ID2CATEGORY.get(gt, "Unknown")

    total += 1
    class_total[category] += 1

    if gt == pred:
        correct += 1
        class_correct[category] += 1

# =========================
# 7. 输出结果
# =========================
print("=" * 60)
print(f"Overall ACC: {correct}/{total} = {correct / total:.4f}")
print("=" * 60)

for cat in sorted(class_total.keys()):
    acc = class_correct[cat] / class_total[cat] if class_total[cat] > 0 else 0
    print(
        f"{cat:10s} | "
        f"ACC = {class_correct[cat]:3d}/{class_total[cat]:3d} "
        f"= {acc:.4f}"
    )
