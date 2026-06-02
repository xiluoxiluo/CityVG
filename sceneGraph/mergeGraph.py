import json

# ----------------------------------------------------
# 输入文件路径
# ----------------------------------------------------
SCENEGRAPH_CAP = "/home/zjj/Code/CityVG/scenegraph_with_descriptions.json"
GRAPH_JSON = "/home/zjj/Code/CityVG/data/birmingham_block_4_graph.json"
OUTPUT = "/home/zjj/Code/CityVG/results/birmingham_block_4_graph.json"


# ----------------------------------------------------
# 工具：统一 caption 为 list[str]
# ----------------------------------------------------
def to_list_of_strings(cap):
    """
    确保 caption 最终是 list[str]
    """
    if cap is None:
        return []

    # string → [string]
    if isinstance(cap, str):
        return [cap]

    # list → flatten 为 string
    if isinstance(cap, list):
        out = []
        for c in cap:
            if isinstance(c, str):
                out.append(c)
            else:
                out.append(str(c))
        return out

    # others → [string]
    return [str(cap)]


# ----------------------------------------------------
# 读取 scenegraph caption
# ----------------------------------------------------
with open(SCENEGRAPH_CAP, "r", encoding="utf-8") as f:
    sg = json.load(f)

edge_caps = {}   # target_id → list[str]

for e in sg["edges"]:
    from_id = str(e["from"])
    cap = e.get("caption", "")
    if cap:
        edge_caps[from_id] = to_list_of_strings(cap)

print(f"Loaded {len(edge_caps)} captions from scenegraph.")


# ----------------------------------------------------
# 读取 birmingham_block_4_graph.json
# ----------------------------------------------------
with open(GRAPH_JSON, "r", encoding="utf-8") as f:
    items = json.load(f)

print(f"Loaded {len(items)} items from {GRAPH_JSON}")


# ----------------------------------------------------
# 替换 caption
# target_id 是字符串，需要与 from_id 做相等判断
# ----------------------------------------------------
updated = []
replace_count = 0

for item in items:
    tid = str(item.get("target_id", ""))

    if tid in edge_caps:
        # 设置 caption 为 list[str]（自动处理）
        item["caption"] = edge_caps[tid]
        replace_count += 1

    updated.append(item)

print(f"Replaced {replace_count} captions.")


# ----------------------------------------------------
# 写回输出文件
# ----------------------------------------------------
with open(OUTPUT, "w", encoding="utf-8") as f:
    json.dump(updated, f, ensure_ascii=False, indent=2)

print(f"Saved updated file → {OUTPUT}")
