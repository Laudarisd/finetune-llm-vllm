import json
import argparse
import os
from pathlib import Path

import torch
import yaml
from PIL import Image, ImageDraw
from transformers import AutoProcessor, AutoModelForImageTextToText

PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = Path(os.environ.get("CONFIG_FILE", PROJECT_ROOT / "config.yaml"))
if not CONFIG_PATH.is_absolute():
    CONFIG_PATH = PROJECT_ROOT / CONFIG_PATH
with CONFIG_PATH.open("r", encoding="utf-8") as config_file:
    CONFIG = yaml.safe_load(config_file)
MODEL_PATH = PROJECT_ROOT / CONFIG["model"]["merged_directory"]
MODEL_MAX_SIDE = int(CONFIG["data"]["max_image_side"])


def load_model():
    """Load the configured merged model and processor for inference."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Merged model not found at {MODEL_PATH}. Train the model or update config.yaml."
        )
    loaded_processor = AutoProcessor.from_pretrained(MODEL_PATH)
    loaded_model = AutoModelForImageTextToText.from_pretrained(
        MODEL_PATH,
        torch_dtype="auto",
        device_map="auto",
    ).eval()
    model_device = next(loaded_model.parameters()).device
    return loaded_model, loaded_processor, model_device


def make_prompt(question: str) -> str:
    return (
        "ONLY output valid JSON.\n"
        "NO explanation.\n"
        "NO thinking.\n"
        "Start with { and end with }.\n"
        "If nothing found, return empty polygons.\n\n"
        'FORMAT: {"class":"object","count":0,"polygons":[],"present":false}\n\n'
        f"QUESTION: {question}\n"
        "JSON:"
    )


def get_model_space_size(orig_w: int, orig_h: int, max_side: int = 1024) -> tuple[int, int]:
    """
    Estimate resized image size used by the model pipeline.
    This is used only when overlay needs model-space -> image-space scaling.
    """
    longest = max(orig_w, orig_h)

    if longest <= max_side:
        return orig_w, orig_h

    ratio = max_side / float(longest)
    new_w = int(round(orig_w * ratio))
    new_h = int(round(orig_h * ratio))
    return new_w, new_h


def infer(
    image_path: str | Path,
    question: str,
    model,
    processor,
    device,
    object_class: str = "object",
    model_max_side: int = MODEL_MAX_SIDE,
):
    image_path = Path(image_path)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size
    model_w, model_h = get_model_space_size(orig_w, orig_h, max_side=model_max_side)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": make_prompt(question)},
            ],
        }
    ]

    prompt = processor.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = processor(
        text=[prompt],
        images=[[image]],
        return_tensors="pt",
        padding=True,
    )

    inputs = {
        k: v.to(device) if hasattr(v, "to") else v
        for k, v in inputs.items()
    }

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=500,
            do_sample=False,
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )

    new_tokens = output[:, inputs["input_ids"].shape[1]:]
    text = processor.decode(new_tokens[0], skip_special_tokens=True).strip()

    if "{" in text and "}" in text:
        text = text[text.find("{"): text.rfind("}") + 1]

    try:
        result = json.loads(text)
    except Exception:
        result = {
            "class": object_class,
            "count": 0,
            "polygons": [],
            "present": False,
        }

    return result, (orig_w, orig_h), (model_w, model_h)


def build_overlay_from_model_json(result: dict, image: Image.Image, model_space_size: tuple[int, int]):
    """Draw predicted polygons, scaling model-space points to original pixels."""
    overlay = image.copy().convert("RGB")
    draw = ImageDraw.Draw(overlay)
    original_w, original_h = overlay.size
    model_w, model_h = model_space_size
    scale_x = original_w / max(model_w, 1)
    scale_y = original_h / max(model_h, 1)

    polygons = list(result.get("polygons", []))
    polygons.extend(
        item.get("polygon", [])
        for item in result.get("annotations", [])
        if isinstance(item, dict)
    )
    for polygon in polygons:
        if not isinstance(polygon, list) or len(polygon) < 3:
            continue
        points = [
            (int(float(point[0]) * scale_x), int(float(point[1]) * scale_y))
            for point in polygon
            if isinstance(point, (list, tuple)) and len(point) >= 2
        ]
        if len(points) >= 3:
            draw.polygon(points, outline=(30, 210, 160), width=3)
    return overlay


def main():
    parser = argparse.ArgumentParser(description="Ask a fine-tuned model about a floor-plan image.")
    parser.add_argument("--image", default="raw_data/images/00001.png", help="Image path")
    parser.add_argument(
        "--question",
        default="list all annotation polygons for hallway",
        help="Question to ask about the floor plan",
    )
    parser.add_argument("--object-class", default="object", help="Fallback class for invalid JSON")
    parser.add_argument(
        "--output",
        default="outputs/overlay_result.png",
        help="Path where the polygon overlay image will be saved",
    )
    args = parser.parse_args()

    model, processor, device = load_model()
    image_path = Path(args.image)
    if not image_path.is_absolute():
        image_path = PROJECT_ROOT / image_path
    result, image_size, model_space_size = infer(
        image_path=image_path,
        question=args.question,
        object_class=args.object_class,
        model=model,
        processor=processor,
        device=device,
    )

    print("MODEL JSON RESULT:")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    print("IMAGE SIZE:", image_size)
    print("MODEL SPACE SIZE:", model_space_size)

    image = Image.open(image_path).convert("RGB")

    overlay_image = build_overlay_from_model_json(
        result,
        image,
        model_space_size=model_space_size,
    )

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = PROJECT_ROOT / output_path
    output_path.parent.mkdir(parents=True, exist_ok=True)
    overlay_image.save(output_path)

    print(f"\nOverlay image saved to: {output_path}")


if __name__ == "__main__":
    main()
