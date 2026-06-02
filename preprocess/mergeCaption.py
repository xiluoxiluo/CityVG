import json
from collections import defaultdict

# === 文件路径 ===
input_path = "/home/zjj/Code/CityVG/data/cityrefer_block/birmingham_block_4.json"
output_path = "/home/zjj/Code/CityVG/visualization/birmingham_block_4_merged.json"

# === 读取原文件 ===
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

# === 按 target_id 聚合 ===
merged = {}
for item in data:
    tid = item["object_id"]
    if tid not in merged:
        merged[tid] = {
            "scene_id": item["scene_id"],
            "object_id": tid,
            "object_name": item["object_name"],
            "description": []
        }
    merged[tid]["description"].append(item["description"])

# === 转换为列表并保存 ===
merged_list = list(merged.values())
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(merged_list, f, indent=2, ensure_ascii=False)

print(f"✅ 合并完成，共处理 {len(data)} 条记录，输出 {len(merged_list)} 个对象。")
print(f"📁 已保存到: {output_path}")
