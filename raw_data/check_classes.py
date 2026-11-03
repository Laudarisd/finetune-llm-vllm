import json
from pathlib import Path

json_dir = Path("raw_data/json")
class_names = set()

for json_file in json_dir.glob("*.json"):
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for shape in data.get("shapes", []):
        label = shape.get("label")
        if label:
            class_names.add(label)

print("Unique class names:")
for name in sorted(class_names):
    print(name)

print(f"\nTotal unique classes: {len(class_names)}")


