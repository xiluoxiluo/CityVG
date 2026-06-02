import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument("--SCENE_ID", type=str, required=True)
args = parser.parse_args()
SCENE_ID = args.SCENE_ID

input_json = r"/home/zjj/Code/CityVG/data/cityrefer_inference/CityRefer_val_ND.json"
output_json = f"/home/zjj/Code/CityVG/data/cityrefer_block/{SCENE_ID}.json"

with open(input_json, "r", encoding="utf-8") as f:
    data = json.load(f)

filtered = [item for item in data if item.get("scene_id") == SCENE_ID]

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(filtered, f, indent=2, ensure_ascii=False)

print(f"✅Total records: {len(filtered)}")
print(f"\n💾 Saved →  {output_json}")

print("****************************************")
print("STEP 1/9: Extract block-level metadata from CityRefer annotations")
print("****************************************")

