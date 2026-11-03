#!/usr/bin/env python3
"""Download a model listed in config.yaml or a model supplied on the CLI."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from huggingface_hub import snapshot_download


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(config_path: Path) -> dict:
    """Read model defaults from the repository's central YAML config."""
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download a Hugging Face model snapshot into a local directory."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=PROJECT_ROOT / "config.yaml",
        help="YAML config supplying model.repository and model.directory.",
    )
    parser.add_argument("--model-id", help="Override the model repository ID.")
    parser.add_argument(
        "--destination",
        type=Path,
        help="Override the local download directory.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    model_id = args.model_id or config["model"]["repository"]
    destination = args.destination or Path(config["model"]["directory"])
    if not destination.is_absolute():
        destination = PROJECT_ROOT / destination
    destination.mkdir(parents=True, exist_ok=True)

    print(f"Downloading {model_id}")
    print(f"Destination: {destination}")
    print("Check the model license/access terms and available disk space first.")
    snapshot_download(repo_id=model_id, local_dir=str(destination))
    print("Download complete.")


if __name__ == "__main__":
    main()
