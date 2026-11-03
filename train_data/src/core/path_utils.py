"""
Simple path and image-metadata helpers for dataset generation.
"""

from pathlib import Path


def normalize_relative_path(path_value: str) -> Path:
    """Normalize path slashes into a Path object."""
    return Path(path_value.replace("\\", "/"))


def match_image_by_stem(json_path: Path, image_dir: Path) -> Path | None:
    """Find image file with the same stem as the JSON file."""
    candidates = []
    for image_path in image_dir.iterdir():
        if image_path.is_file() and image_path.stem.lower() == json_path.stem.lower():
            candidates.append(image_path)
    return sorted(candidates)[0] if candidates else None


def resolve_image_path_from_json(data: dict, json_path: Path, image_dir: Path) -> Path | None:
    """Resolve image path from JSON metadata first, then stem match fallback."""
    image_path_value = data.get("imagePath")
    if image_path_value:
        candidate = (json_path.parent / normalize_relative_path(image_path_value)).resolve()
        if candidate.is_file():
            return candidate
    return match_image_by_stem(json_path, image_dir)


def build_image_metadata(data: dict, image_path: Path) -> dict:
    """Build simple image metadata dictionary."""
    metadata = {
        "image_path": str(image_path),
        "coordinate_system": "pixel_xy",
    }

    width = data.get("imageWidth")
    height = data.get("imageHeight")

    if isinstance(width, int) and isinstance(height, int):
        metadata["image_size"] = {"width": width, "height": height}

    return metadata
