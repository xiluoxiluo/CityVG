#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import json
import torch
from tqdm import tqdm
from sentence_transformers import SentenceTransformer, util

def generate_vis_candidates(
    scene_name: str,
    caption: str,
    merged_json: str,
    out_json: str,
    topk: int = 10,
    progress=None
):
    """直接用 caption 与 item['description'] 做相似度匹配"""

    # === 定义安全的进度回调 ===
    def safe_progress(value, desc=""):
        if progress is not None:
            try:
                progress(value, desc=desc)
            except Exception:
                pass
        else:
            print(f"[{value*100:.1f}%] {desc}")

    # ======================================================
    # 1. Load merged.json
    # ======================================================
    safe_progress(0.0, desc=f"📥 Loading merged JSON for {scene_name} ...")
    with open(merged_json, "r", encoding="utf-8") as f:
        merged = json.load(f)

    # ======================================================
    # 2. Load BGE
    # ======================================================
    safe_progress(0.1, desc="🧠 Encoding corpus embeddings ...")
    model = SentenceTransformer("BAAI/bge-large-en-v1.5")

    corpus_texts = [" ".join(item["description"]) for item in merged]
    id_map = [item["object_id"] for item in merged]

    with torch.inference_mode():
        corpus_emb = model.encode(
            corpus_texts,
            convert_to_tensor=True,
            normalize_embeddings=True
        )

    # ======================================================
    # 3. Query embedding（直接用 caption）
    # ======================================================
    safe_progress(0.5, desc="🔍 Computing caption similarity ...")
    with torch.inference_mode():
        q_emb = model.encode(
            caption,
            convert_to_tensor=True,
            normalize_embeddings=True
        )
        sims = util.cos_sim(q_emb, corpus_emb)[0]

    topk_vals, topk_idx = torch.topk(sims, k=min(topk, len(merged)))
    candidates = [id_map[int(i)] for i in topk_idx]

    # ======================================================
    # 4. Save result
    # ======================================================
    results = [{
        "scan_id": scene_name,
        "caption": caption,
        "candidates": candidates
    }]

    os.makedirs(os.path.dirname(out_json), exist_ok=True)
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    safe_progress(1.0, desc="✅ Caption-based candidate JSON generated.")
    print(f"✅ Saved → {out_json}")

    return out_json


if __name__ == "__main__":
    scene = "birmingham_block_4"
    caption = "The outer section of a large building on Birchfield Road's corner of the map has a white and blue roof."
    merged = f"/home/zjj/Code/CityVG/visualization/{scene}_merged.json"
    out = f"/home/zjj/Code/CityVG/visualization/vis_{scene}_candidates.json"
    generate_vis_candidates(scene, caption, merged, out, topk=10)
