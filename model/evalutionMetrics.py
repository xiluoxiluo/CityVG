#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
CityVG Evaluation Metrics (with Progress Bar + Dynamic IoU)
------------------------------------------------------------
输入:
  1) results/qwen3vl_plus_grounding_birmingham.json
  2) /home/zjj/Code/CityAnchor/data/data_cityrefer/IoU_File/birmingham_block_4_iou.json
  3) 对应场景点云文件：
     /home/zjj/Code/CityVG/data/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250/<scan_id>.pth
"""

import json
import os
import torch
import numpy as np
from tqdm import tqdm
from collections import defaultdict


# ==============================
# 1️⃣ 文件路径
# ==============================
RESULT_JSON = "results/qwen3vl_plus_grounding_birmingham.json"
IOU_JSON = "/home/zjj/Code/CityAnchor/data/data_cityrefer/IoU_File/birmingham_block_4_iou.json"
PTH_ROOT = "/home/zjj/Code/CityVG/data/data_cityrefer/sensaturban/pointgroup_data/balance_split/random-50_crop-250"

# ==============================
# 2️⃣ IoU计算函数
# ==============================
def compute_bbox_iou(box1, box2):
    """计算两个3D包围盒的IoU ([x, y, z, dx, dy, dz])"""
    min1 = box1[:3] - box1[3:] / 2
    max1 = box1[:3] + box1[3:] / 2
    min2 = box2[:3] - box2[3:] / 2
    max2 = box2[:3] + box2[3:] / 2

    inter_min = np.maximum(min1, min2)
    inter_max = np.minimum(max1, max2)
    inter_size = np.maximum(0, inter_max - inter_min)
    inter_vol = np.prod(inter_size)

    vol1 = np.prod(box1[3:])
    vol2 = np.prod(box2[3:])
    union_vol = vol1 + vol2 - inter_vol
    return 0.0 if union_vol == 0 else inter_vol / union_vol


def get_iou_from_pth(pth_path, gtid, predid):
    """从 pth 文件根据 gtid 和 predid 计算 IoU，支持 tuple 格式"""
    try:
        data = torch.load(pth_path, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"\n⚠️ Failed to load {pth_path}: {e}")
        return 0.0

    # 兼容 tuple 或 dict 格式
    if isinstance(data, (tuple, list)) and len(data) >= 7:
        instance_bboxes = data[6]  # 第7项是 instance_bboxes
    elif isinstance(data, dict):
        instance_bboxes = data.get("instance_bboxes", [])
    else:
        print(f"\n⚠️ Unexpected structure in {pth_path}")
        return 0.0

    # 转为 numpy
    if isinstance(instance_bboxes, dict):
        instance_bboxes = list(instance_bboxes.values())
    instance_bboxes = np.array(instance_bboxes)
    if instance_bboxes.ndim == 1:
        instance_bboxes = np.expand_dims(instance_bboxes, axis=0)

    bbox_gt = bbox_pred = None
    for box in instance_bboxes:
        if len(box) < 7:
            continue
        if int(box[-1]) == int(gtid):
            bbox_gt = box
        if int(box[-1]) == int(predid):
            bbox_pred = box

    if bbox_gt is None or bbox_pred is None:
        return 0.0
    return compute_bbox_iou(bbox_gt[:6], bbox_pred[:6])


# ==============================
# 3️⃣ 加载数据
# ==============================
with open(RESULT_JSON, "r", encoding="utf-8") as f:
    results = json.load(f)
with open(IOU_JSON, "r", encoding="utf-8") as f:
    iou_data = json.load(f)

iou_data = {int(k): v[0] for k, v in iou_data.items()}
total_count = len(results)
print(f"📦 Loaded {total_count} prediction results.\n")

# ==============================
# 4️⃣ 初始化计数器
# ==============================
hit1 = hit5 = 0
rank_1_count = rank_le_2_count = rank_le_3_count = rank_le_5_count = rank_le_10_count = 0
rank_1_acc25_count = rank_2_acc25_count = rank_3_acc25_count = 0
rank_1_acc50_count = rank_2_acc50_count = rank_3_acc50_count = 0
rank_5_acc50_count = rank_10_acc50_count = 0
class_acc = defaultdict(lambda: {"total": 0, "acc50": 0})


# ==============================
# 5️⃣ 主评估循环 (带进度条)
# ==============================
for item in tqdm(results, desc="🔍 Evaluating CityVG Results", ncols=100):
    gtid = str(item.get("GTID"))
    pred = str(item.get("best_id"))
    cands = [str(x) for x in item.get("candidates", [])]
    obj_name = item.get("obj_name", "Unknown")
    scan_id = item.get("scan_id", "birmingham_block_4")

    # 动态 IoU 计算
    if gtid == pred:
        iou = iou_data.get(int(gtid), 0.0)
    else:
        pth_path = os.path.join(PTH_ROOT, f"{scan_id}.pth")
        iou = get_iou_from_pth(pth_path, gtid, pred) if os.path.exists(pth_path) else 0.0

    iou_object = [iou]

    # === Hit@K ===
    if pred == gtid:
        hit1 += 1
    if gtid in cands[:5]:
        hit5 += 1

    # === Rank ===
    rank = cands.index(pred) + 1 if pred in cands else len(cands) + 1

    # Rank == 1
    if rank == 1:
        rank_1_count += 1
        class_acc[obj_name]["total"] += 1
        if iou >= 0.25:
            rank_1_acc25_count += 1
        if iou >= 0.50:
            rank_1_acc50_count += 1
            class_acc[obj_name]["acc50"] += 1

    # Rank ≤ 2
    if rank <= 2:
        rank_le_2_count += 1
        if iou >= 0.25:
            rank_2_acc25_count += 1
        if iou >= 0.50:
            rank_2_acc50_count += 1

    # Rank ≤ 3
    if rank <= 3:
        rank_le_3_count += 1
        if iou >= 0.25:
            rank_3_acc25_count += 1
        if iou >= 0.50:
            rank_3_acc50_count += 1

    # Rank ≤ 5
    if rank <= 5:
        rank_le_5_count += 1
        if iou >= 0.50:
            rank_5_acc50_count += 1

    # Rank ≤ 10
    if rank <= 10:
        rank_le_10_count += 1
        if iou >= 0.50:
            rank_10_acc50_count += 1


# ==============================
# 6️⃣ 辅助函数
# ==============================
def ratio(a, b):
    return a / b if b > 0 else 0


# ==============================
# 7️⃣ 输出评估结果
# ==============================
print("\n========== 🧮 Overall Evaluation ==========")
print(f"Total Samples: {total_count}")
print(f"🎯 Hit@1 = {hit1}/{total_count} = {hit1/total_count:.2%}")
print(f"🎯 Hit@5 = {hit5}/{total_count} = {hit5/total_count:.2%}")
print(f"Sample of Acc@0.00(Rank 1): {rank_1_count}, Ratio: {ratio(rank_1_count,total_count):.2%}")
print(f"Sample of Acc@0.25(Rank 1): {rank_1_acc25_count}, Ratio: {ratio(rank_1_acc25_count,total_count):.2%}")
print(f"Sample of Acc@0.50(Rank 1): {rank_1_acc50_count}, Ratio: {ratio(rank_1_acc50_count,total_count):.2%}")
print(f"Sample of Acc@0.50(Rank 2): {rank_2_acc50_count}, Ratio: {ratio(rank_2_acc50_count,total_count):.2%}")
print(f"Sample of Acc@0.50(Rank 3): {rank_3_acc50_count}, Ratio: {ratio(rank_3_acc50_count,total_count):.2%}")
print(f"Sample of Acc@0.50(Rank 5): {rank_5_acc50_count}, Ratio: {ratio(rank_5_acc50_count,total_count):.2%}")
print(f"Sample of Acc@0.50(Rank 10): {rank_10_acc50_count}, Ratio: {ratio(rank_10_acc50_count,total_count):.2%}")
print("==========================================\n")

print("========== 🏷️ Per-Class Rank@1 IoU≥0.5 ==========")
for cname, stats in class_acc.items():
    acc_ratio = ratio(stats["acc50"], stats["total"])
    print(f"{cname:<10}: {stats['acc50']:>4}/{stats['total']:<4} = {acc_ratio:.2%}")
print("=================================================\n")


# ==============================
# 8️⃣ 保存汇总结果
# ==============================
summary_path = "results/eval_summary_birmingham.json"
summary = {
    "Total_Samples": total_count,
    "Hit@1": ratio(hit1, total_count),
    "Hit@5": ratio(hit5, total_count),
    "Rank1@0.25": ratio(rank_1_acc25_count, total_count),
    "Rank1@0.50": ratio(rank_1_acc50_count, total_count),
    "Rank2@0.50": ratio(rank_2_acc50_count, total_count),
    "Rank3@0.50": ratio(rank_3_acc50_count, total_count),
    "Rank5@0.50": ratio(rank_5_acc50_count, total_count),
    "Rank10@0.50": ratio(rank_10_acc50_count, total_count),
    "PerClass@0.50": {c: ratio(v["acc50"], v["total"]) for c, v in class_acc.items()}
}
os.makedirs("results", exist_ok=True)
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2, ensure_ascii=False)

print(f"💾 Results saved → {summary_path}")
print("✅ Evaluation Completed.\n")
