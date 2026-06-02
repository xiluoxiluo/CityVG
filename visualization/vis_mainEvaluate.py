#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vis_mainEvaluate.py
------------------------------------
封装 mainEvaluate.py 的核心 grounding 逻辑
  输入: vis_candidates.json
  输出: vis_results.json
"""

import json
from tqdm import tqdm
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from model.mainEvaluate import vlm_grounding_api


def run_vis_grounding(scene_name: str, candidates_json: str, img_root: str, out_json: str, progress=None):
    """基于 mainEvaluate 的完整 grounding 推理流程"""

    # === 安全包装函数，防止 progress=None 报错 ===
    def safe_progress(value, desc=""):
        if progress is not None:
            try:
                progress(value, desc=desc)
            except Exception:
                pass
        else:
            # 控制台打印简易进度
            print(f"[{value*100:.1f}%] {desc}")

    with open(candidates_json, "r", encoding="utf-8") as f:
        items = json.load(f)

    safe_progress(0, desc=f"🧠 Running grounding for {scene_name} ...")
    results = []
    total = len(items)
    for idx, item in enumerate(tqdm(items, desc=f"Grounding ({scene_name})", ncols=100)):
        caption, cands = item["caption"], item["candidates"]
        try:
            res = vlm_grounding_api(caption, cands, img_root)
            item["predicted_id"] = res.get("best_id", None)
            item["reason"] = res.get("reason", "")
        except Exception as e:
            print("⚠️ grounding error:", e)
            item["predicted_id"] = None
            item["reason"] = str(e)
        results.append(item)
        safe_progress((idx + 1) / total, desc=f"🔹 Inference {idx + 1}/{total} ...")

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    safe_progress(1.0, desc="✅ Grounding inference completed.")
    print(f"✅ Saved → {out_json}")
    return out_json


if __name__ == "__main__":
    scene = "birmingham_block_4"
    cand_json = f"/home/zjj/Code/CityVG/visualization/vis_{scene}_candidates.json"
    img_root = f"/home/zjj/Code/CityVG/data/cityrefer_preprocessed/{scene}"
    out_json = f"/home/zjj/Code/CityVG/visualization/vis_{scene}_results.json"
    run_vis_grounding(scene, cand_json, img_root, out_json)
