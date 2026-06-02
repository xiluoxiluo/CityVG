#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Generate a scene graph (nodes + edges) from a Birmingham bbox.json file.
Edges:
    - For every landmark: generate self-loop edge
    - For every object:   connect to nearest landmark

relation field kept empty "" (for later writing)
"""

import os
import json
import math
import argparse
from pathlib import Path
import argparse

def load_bboxes(json_path):
    """Load bbox json and return list of object dicts."""
    json_path = Path(json_path)
    with json_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "bboxes" in data:
        return data["bboxes"]

    if isinstance(data, list):
        return data

    raise ValueError(f"Unsupported bbox JSON format: {json_path}")

def build_scene_graph(bboxes):
    """
    Build scene graph:
    - Nodes: all objects
    - Edges:
        * Landmark → Landmark (self-loop)
        * Object → nearest landmark
    """

    nodes = []
    centers = {}

    landmarks = []   # (id, cx, cy)
    objects = []     # (id, cx, cy)

    for obj in bboxes:
        obj_id = obj["object_id"]
        obj_name = obj.get("object_name", "")
        landmark = obj.get("landmark", "")
        bbox = obj["bbox"]

        cx, cy = bbox[0], bbox[1]

        node = {
            "id": obj_id,
            "object_name": obj_name,
            "landmark_name": landmark,
            "is_landmark": (landmark != ""),
            "center": [cx, cy]
        }

        nodes.append(node)
        centers[obj_id] = (cx, cy)

        if landmark != "":
            landmarks.append((obj_id, cx, cy))
        else:
            objects.append((obj_id, cx, cy))

    edges = []

    for lm_id, lx, ly in landmarks:
        edges.append({
            "from": lm_id,
            "to": lm_id,
            "relation": "",
            "distance": 0.0
        })

    if len(landmarks) == 0:
        print("⚠ Warning: No landmark found in this scene.")
        return {"nodes": nodes, "edges": edges}

    for obj_id, ox, oy in objects:
        nearest_lm = None
        nearest_dist = float("inf")

        for lm_id, lx, ly in landmarks:
            d = math.sqrt((ox - lx)**2 + (oy - ly)**2)
            if d < nearest_dist:
                nearest_lm = lm_id
                nearest_dist = d

        edges.append({
            "from": obj_id,
            "to": nearest_lm,
            "relation": "",
            "distance": nearest_dist
        })

    return {
        "nodes": nodes,
        "edges": edges
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--SCENE_ID", type=str, required=True)
    args = parser.parse_args()
    SCENE_ID = args.SCENE_ID
    json_path = os.path.join(
        "/home/zjj/Code/CityVG/data/cityrefer_bbox/box3d",
        f"{SCENE_ID}_bbox.json"
    )
    out_path = os.path.join(
        "/home/zjj/Code/CityVG/data/cityrefer_graph",
        f"{SCENE_ID}_graph.json"
    )

    bboxes = load_bboxes(json_path)
    print(f"✔ Loaded {len(bboxes)} objects.")

    print("🧱 Building scene graph (landmark self-loop + object→nearest landmark)...")
    graph = build_scene_graph(bboxes)

    out_path = Path(out_path)
    print(f"💾 save to: {out_path}")

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(graph, f, indent=2)


    print("****************************************")
    print("STEP 5/9: Build scene graph from instance-level geometric relations")
    print("****************************************")

if __name__ == "__main__":
    main()
