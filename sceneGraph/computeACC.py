#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from tqdm import tqdm

# ============================================================
# Config
# ============================================================
JSON_PATH = "/home/zjj/Code/CityVG/data/cityrefer_inference/birmingham_block_5_inference.json"
K = 5                                        # Hit@K
PRINT_DETAIL = False                  # 是否打印每条样本

# ============================================================
# Load File
# ============================================================
with open(JSON_PATH, "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"📦 Loaded {len(items)} items from {JSON_PATH}")

results = []
hit1, hitk = 0, 0

# ============================================================
# Evaluation Loop
# ============================================================
for item in tqdm(items, desc="📊 Evaluating", ncols=100):

    cap = item["caption"]
    cands = item["candidates"]
    pred = item["best_id"]              # 从模型预测结果中读取
    gtid = item.get("GTID")

    if PRINT_DETAIL:
        print("\n" + "=" * 80)
        print("📝 Caption:", cap)
        print("📍 Candidates:", cands)
        print("🎯 GT:", gtid)
        print("🤖 Pred:", pred)
        print("🧠 Reason:", item.get("reason", ""))

    # =========================
    #     Hit@1 统计
    # =========================
    if gtid:
        if pred == gtid:
            hit1 += 1
            if PRINT_DETAIL:
                print("✅ Hit@1 Correct!")
        else:
            if PRINT_DETAIL:
                print("❌ Hit@1 Wrong.")

        # =========================
        #     Hit@K 统计
        # =========================
        if gtid in cands[:K]:
            hitk += 1

# ============================================================
# Final Output
# ============================================================
total = len(items)
hit1_rate = hit1 / total
hitk_rate = hitk / total

print("\n" + "=" * 60)
print(f"🎯 Hit@1 = {hit1_rate:.3f}   ({hit1}/{total})")
print(f"🎯 Hit@{K} = {hitk_rate:.3f}   ({hitk}/{total})")
print("=" * 60)
