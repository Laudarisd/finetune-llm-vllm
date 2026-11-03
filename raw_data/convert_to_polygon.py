import os
import json
import random
from PIL import Image
import shutil

# ===========================
# CONFIG
# ===========================
IMAGES_DIR = "data/images"
OBB_DIR = "data/labels"
SEG_DIR = "data/seg"  # optional, for backup JSON
LLM_JSONL = "data/llm_train.jsonl"

os.makedirs(SEG_DIR, exist_ok=True)

# Class mapping
CLASS_NAMES = [
    "junc_I", "junc_L", "junc_T", "junc_X",
    "door_normal", "door_double", "door_hinged", "window",
    "basin", "bathtub", "commode", "gas",
    "sink", "elevator", "stairs", "sign",
    "symbol", "symbol_text"
]

# Prompt variations for LLM instruction tuning
PROMPTS = [
    "Detect all floor plan entities such as walls, doors, and windows. Return the result as JSON with entity type and polygon coordinates.",
    "Find all doors in the given floor plan image and return polygons only.",
    "Extract all junctions and doors in JSON format from this floor plan.",
    "Identify walls, doors, and windows in the image and return polygons.",
    "Detect floor plan structural elements and output JSON with type and polygon."
]

# ===========================
# HELPER FUNCTIONS
# ===========================
def obb_line_to_polygon(line, img_w, img_h):
    """Convert one line from OBB TXT to class name + absolute polygon"""
    parts = line.strip().split()
    cls_id = int(parts[0])
    coords = list(map(float, parts[1:]))  # normalized 0-1
    polygon = []
    for i in range(0, 8, 2):
        x = coords[i] * img_w
        y = coords[i+1] * img_h
        polygon.append([round(x, 2), round(y, 2)])
    return CLASS_NAMES[cls_id], polygon

# ===========================
# MAIN PROCESS
# ===========================
with open(LLM_JSONL, "w") as out_file:
    for txt_file in os.listdir(OBB_DIR):
        if not txt_file.endswith(".txt"):
            continue

        base_name = os.path.splitext(txt_file)[0]
        img_file = os.path.join(IMAGES_DIR, base_name + ".png")
        if not os.path.exists(img_file):
            print(f"⚠️ Image not found for {txt_file}, skipping.")
            continue

        img = Image.open(img_file)
        w, h = img.size

        # Read OBB TXT
        shapes = []
        with open(os.path.join(OBB_DIR, txt_file)) as f:
            for line in f:
                cls_name, polygon = obb_line_to_polygon(line, w, h)
                shapes.append({
                    "label": cls_name,
                    "polygon": polygon
                })

        # Optional: save backup polygon JSON in SEG_DIR
        seg_json_path = os.path.join(SEG_DIR, f"{base_name}_obb.json")
        seg_json = {
            "image": base_name + ".png",
            "annotations": shapes
        }
        with open(seg_json_path, "w") as f:
            json.dump(seg_json, f, indent=2)

        # Build LLM JSONL sample
        prompt_text = random.choice(PROMPTS)
        output = {
            "image_width": w,
            "image_height": h,
            "entities": [
                {"type": s["label"], "polygon": s["polygon"]}
                for s in shapes
            ]
        }

        sample = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_file},
                        {"type": "text", "text": prompt_text}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": json.dumps(output)}
                    ]
                }
            ]
        }

        out_file.write(json.dumps(sample) + "\n")

print(f"✅ All OBB TXT converted to LLM JSONL at {LLM_JSONL}")
