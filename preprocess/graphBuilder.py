import os
import re
import json
from collections import defaultdict
from difflib import SequenceMatcher
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# ==============================================================
#                   Qwen3-8B 文本关系抽取器
# ==============================================================

class Qwen3TextRelationExtractor:
    def __init__(self, model_name="Qwen/Qwen3-8B"):
        print(f"🚀 Loading {model_name} ...")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto"
        )

    def _chat(self, prompt, max_new_tokens=2048):
        """Use chat_template to talk with Qwen3-8B in pure text mode."""
        messages = [
            {"role": "user", "content": prompt}
        ]

        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False
        )

        inputs = self.tokenizer([text], return_tensors="pt").to(self.model.device)

        output_ids = self.model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
        )[0]

        gen = output_ids[len(inputs.input_ids[0]):]
        decoded = self.tokenizer.decode(gen, skip_special_tokens=True)

        return decoded

    def extract_relations(self, caption: str):
        """
        输出格式：
        [
          {
            "subject": {"text": "...", "color": "...", "category": "..."},
            "relation": "...",
            "object": "LANDMARK NAME"
          }
        ]
        """

        prompt = f"""
Extract spatial relations from the caption, following these STRICT RULES:

==========================
### SUBJECT:
- Subject may be an object like "white car", "blue van", "red truck".
- Extract:
    "text"     (full phrase),
    "color"    (white/blue/etc),
    "category" (car/van/etc)

==========================
### OBJECT (LANDMARKS ONLY):
A landmark is a proper noun referring to:
- building, shop, church, institution, company
- road name (“Station Bridge Road”)
- intersection (“Birchfield Road and Wellington Road”)
- anything capitalized and location-like

❌ DO NOT treat "blue car", "white car", "car", etc. as landmarks.
❌ IGNORE any relation where object is NOT a landmark.

==========================
### RELATIONS:
Include only spatial relations:
behind, in front of, next to, at, on, across from,
between, facing, near, north of, south of, east of, west of, etc.

==========================
### OUTPUT (STRICT):
Return ONLY valid Python list of dicts:

[
  {{
    "subject": {{"text": "...", "color": "...", "category": "..."}},
    "relation": "...",
    "object": "LANDMARK NAME"
  }}
]

==========================

Caption: "{caption}"
"""

        raw = self._chat(prompt)

        # 尝试抽取 list
        try:
            start = raw.find("[")
            end = raw.rfind("]")
            if start != -1 and end != -1 and end > start:
                json_str = raw[start:end+1]
                data = eval(json_str)
                return data
        except Exception as e:
            print("⚠️ JSON parse error:", e)                                                                                        
            print("RAW:", raw)

        return []


# ==============================================================
#                        工具函数
# ==============================================================

def normalize_lm(name: str) -> str:
    """Landmark 名称归一化，用于去重、匹配"""
    name = name.lower().replace("&", "and")
    name = re.sub(r"\s+", " ", name).strip()
    return name

def fuzzy_equal(a: str, b: str, thr: float = 0.85) -> bool:
    return SequenceMatcher(None, a, b).ratio() >= thr


# ==============================================================
#                      Scene Graph Builder
# ==============================================================

def build_scene_graph_with_qwen(
    bbox_path: str,
    caption_path: str,
    extractor: Qwen3TextRelationExtractor
):
    """
    输入：bbox + captions
    输出：Scene Graph
    """

    # ------------------------------
    # Load JSON files
    # ------------------------------
    with open(bbox_path, "r") as f:
        bbox_data = json.load(f)
    with open(caption_path, "r") as f:
        cap_data = json.load(f)

    # ------------------------------
    # 初始化节点
    # ------------------------------
    nodes = {}
    lm_norm2id = {}

    for obj in bbox_data["bboxes"]:
        oid = obj["object_id"]
        nodes[oid] = {
            "node_id": oid,
            "name": obj["object_name"],
            "landmark_from_bbox": obj["landmark"],
            "type": "object",
        }

        lm_name = obj["landmark"].strip()
        if lm_name:
            norm = normalize_lm(lm_name)
            if norm not in lm_norm2id:
                lm_id = "lm:" + norm.replace(" ", "_")
                lm_norm2id[norm] = lm_id

                nodes[lm_id] = {
                    "node_id": lm_id,
                    "name": lm_name,
                    "norm_name": norm,
                    "type": "landmark"
                }

    # ------------------------------
    # Pass 1: 颜色+类别 → object_id
    # ------------------------------
    color_cat2obj_ids = defaultdict(set)

    print("🔍 Pass 1: building color+category mapping ...")
    for ann in tqdm(cap_data, desc="Pass 1: Extracting subject color/category", ncols=100):
        caption = ann["caption"]
        target_id = int(ann["target_id"])

        triples = extractor.extract_relations(caption)

        for t in triples:
            subj = t["subject"]
            color = subj["color"].lower()
            cat = subj["category"].lower()

            if color and cat:
                color_cat2obj_ids[(color, cat)].add(target_id)


    # ------------------------------
    # Pass 2: 构建 Scene Graph Edges
    # ------------------------------
    edges = []

    print("🧩 Pass 2: building edges ...")
    for ann in tqdm(cap_data, desc="Pass 2: Building edges", ncols=100):
        caption = ann["caption"]
        target_id = int(ann["target_id"])
        scan_id = ann.get("scan_id", "")
        ann_id = ann["ann_id"]

        triples = extractor.extract_relations(caption)

        for t in triples:
            subj = t["subject"]
            rel = t["relation"]
            obj_lm = t["object"]

            # ----------------------
            # Landmark 归一化 + 匹配
            # ----------------------
            norm_obj = normalize_lm(obj_lm)

            matched_norm = None
            for existed in lm_norm2id.keys():
                if fuzzy_equal(norm_obj, existed):
                    matched_norm = existed
                    break

            if matched_norm is not None:
                lm_id = lm_norm2id[matched_norm]
            else:
                lm_id = "lm:" + norm_obj.replace(" ", "_")
                lm_norm2id[norm_obj] = lm_id
                nodes[lm_id] = {
                    "node_id": lm_id,
                    "name": obj_lm,
                    "norm_name": norm_obj,
                    "type": "landmark"
                }

            # ----------------------
            # subject → candidates
            # ----------------------
            color = subj["color"].lower()
            cat = subj["category"].lower()

            src_candidates = []
            if color and cat:
                key = (color, cat)
                if key in color_cat2obj_ids:
                    src_candidates = sorted(list(color_cat2obj_ids[key]))

            if not src_candidates:
                src_candidates = [target_id]

            # ----------------------
            # 写入边
            # ----------------------
            edges.append({
                "src_candidates": src_candidates,
                "subject": subj,
                "dst": lm_id,
                "relation": rel,
                "caption_id": ann_id,
                "scan_id": scan_id
            })

    # ------------------------------
    # 最终 Scene Graph
    # ------------------------------
    scene_graph = {
        "nodes": list(nodes.values()),
        "edges": edges
    }

    return scene_graph


# ==============================================================
#                            MAIN
# ==============================================================

if __name__ == "__main__":
    bbox_path = "/home/zjj/Code/CityVG/data/birmingham_block_4_bbox.json"
    caption_path = "/home/zjj/Code/CityVG/data/birmingham_block_4.json"

    extractor = Qwen3TextRelationExtractor("Qwen/Qwen3-8B")

    graph = build_scene_graph_with_qwen(
        bbox_path=bbox_path,
        caption_path=caption_path,
        extractor=extractor
    )

    save_path = "/home/zjj/Code/CityVG/results/scene_graph_qwen3_8b_birmingham_block_4.json"
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2, ensure_ascii=False)

    print("✅ Scene Graph saved to:", save_path)
