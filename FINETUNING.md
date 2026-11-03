# Fine-Tuning a Vision-Language Model for Floor-Plan Understanding

> Portfolio documentation for this repository. The project fine-tunes Qwen3.5-9B to answer questions about architectural floor-plan images. The current data includes room/object questions, counts and comparisons, junction detection, and polygon-style annotations. This is a visual understanding task; it does not generate new architectural plans.

## Contents

1. [Introduction](#1-introduction)
2. [Project contents](#2-project-contents)
3. [Requirements](#3-requirements)
4. [Goal](#4-goal)
5. [Dataset preparation](#5-dataset-preparation)
6. [Model preparation and fine-tuning concepts](#6-model-preparation-and-fine-tuning-concepts)
7. [Training files and setup](#7-training-files-and-setup)
8. [Training log](#8-training-log)
9. [Output and inference](#9-output-and-inference)
10. [Portfolio images](#10-portfolio-images)

## 1. Introduction

This repository adapts a pretrained vision-language model (VLM) to interpret architectural floor plans. Each example pairs a plan image and a natural-language question with a target answer. Answers can be short text or JSON containing counts, comparisons, object classes, or polygon coordinates.

The main training script loads a local Qwen3.5-9B model, prepares image/text examples, adds LoRA adapters, and runs supervised fine-tuning (SFT). It evaluates and saves checkpoints during training, selects the best checkpoint by validation loss, and merges the adapter into the base model for inference.

## 2. Project contents

| Path | Purpose |
| --- | --- |
| `config.yaml` | Central model, data, hardware, LoRA, and trainer settings |
| `finetune_run.sh` / `tensorboard.sh` | Root-level training and TensorBoard launchers |
| `model_downloader/download.py` | Config-driven Hugging Face model downloader |
| `notebooks/extra/` | Optional analysis and inference notebooks |
| `train_data/vlm_train.jsonl` | Chat-style training examples |
| `train_data/vlm_val.jsonl` | Validation examples |
| `raw_data/images/` | Floor-plan images referenced by examples |
| `train.py` | Main Unsloth/TRL SFT pipeline |
| `requirements.txt` | Python package requirements |
| `models/Qwen3.5-9B/` | Local base model and processor files |
| `outputs/qwen35-floorplan-lora/` | Logs, checkpoints, adapter, and merged model |
| `inference.py` | Inference-related script |

The JSONL data currently contains both relative and absolute image paths. Relative paths (such as `raw_data/images/plan.png`) make the project easier to move; ensure the data loader resolves them against the repository or data root.

## 3. Requirements

### Software and hardware

- Python 3.10 or newer is a practical baseline. Select a version supported by the installed PyTorch, Transformers, and Unsloth builds.
- A CUDA-capable NVIDIA GPU is needed for practical fine-tuning. Memory use depends on image size, sequence length, batch size, and GPU count.
- Install CUDA-enabled PyTorch for the host's CUDA runtime. `requirements.txt` includes a CUDA 12.1 example; choose a compatible PyTorch build.

### Main libraries

Package ranges are listed in [requirements.txt](requirements.txt).

- PyTorch (`torch`, `torchvision`, `torchaudio`) — tensor and GPU computation; install separately as noted in requirements.
- Hugging Face Transformers, Hub, and Tokenizers — model, processor, and tokenizer loading.
- `datasets` — JSONL dataset loading.
- `trl` — supervised fine-tuning trainer and config.
- `peft` — LoRA adapters and merging.
- `unsloth` — optimized vision-language loading and training helpers.
- `accelerate` — distributed and multi-GPU launch.
- `bitsandbytes` — low-bit model loading.
- `qwen-vl-utils`, Pillow, and OpenCV — vision utilities and image processing.
- `tensorboard` — training metrics.

Install PyTorch first, followed by the project requirements:

```bash
# Example only; choose a wheel compatible with your CUDA setup.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

The model is stored locally under `models/Qwen3.5-9B/`. On a fresh machine, obtain it from its authorized Hugging Face repository and update `MODEL_DIR` if storing it elsewhere. Check the model's access terms before redistributing weights.

## 4. Goal

**Goal:** adapt a pretrained VLM to inspect an architectural floor-plan image and answer questions consistently, using structured JSON where appropriate.

Example tasks represented by the data:

- Identify rooms and architectural objects.
- Count or compare classes such as bedrooms, sinks, basins, doors, and windows.
- Detect geometric features such as X-junctions.
- Return object annotations as polygons in image coordinates.

This project does not produce CAD/BIM geometry or provide building-code determinations. Results depend on dataset coverage and label quality.

## 5. Dataset preparation

Each line in the train and validation files is a standalone JSON object. Its `messages` array has a user message with an image and question, followed by an assistant message with the expected answer.

### Example JSONL record

This illustrative record follows the repository schema. Replace the filename with an existing image.

```json
{"messages":[
  {"role":"user","content":[
    {"type":"image","image":"raw_data/images/example_floor_plan.png"},
    {"type":"text","text":"How many bedrooms are shown? Reply as JSON."}
  ]},
  {"role":"assistant","content":[
    {"type":"text","text":"{\"bedroom_count\":2}"}
  ]}
]}
```

A polygon annotation example:

```json
{"messages":[
  {"role":"user","content":[
    {"type":"image","image":"raw_data/images/example_floor_plan.png"},
    {"type":"text","text":"List the doors visible in this plan as class and polygon."}
  ]},
  {"role":"assistant","content":[
    {"type":"text","text":"{\"annotations\":[{\"class\":\"Door\",\"polygon\":[[120,80],[160,80],[160,140],[120,140]]}]}"}
  ]}
]}
```

### Preparation checklist

1. Confirm every referenced image exists and is readable.
2. Keep image labels aligned; define polygon coordinates consistently (the examples use image pixel coordinates).
3. Use consistent class names and answer formats; return valid JSON when JSON is requested.
4. Split by source floor plan/image to reduce near-duplicate leakage into validation.
5. Check malformed records, missing images, duplicates, empty labels, and class imbalance.
6. Keep validation data out of training and use it for evaluation/checkpoint selection.

The local JSONL files contain 1,037,246 training lines and 238,685 validation lines. The recorded run summary reports 1,037,246 selected training samples and 47,737 validation samples (20%). The percentages are set in `config.yaml`; they can be adjusted for a smaller experiment.

## 6. Model preparation and fine-tuning concepts

### Dense and Mixture-of-Experts models

- **Dense model:** each token passes through the same feed-forward blocks in each transformer layer; the full model participates in each forward pass.
- **Mixture of Experts (MoE):** a router selects one or more feed-forward experts for each token. A model can have many total parameters while activating only a subset per token.

Dense and MoE describe model architectures. Fine-tuning does not change one architecture into the other. The training script loads the architecture defined by the selected Qwen checkpoint; check that checkpoint's config/model card before labeling it dense or MoE.

### PEFT, LoRA, and QLoRA

- **PEFT (Parameter-Efficient Fine-Tuning):** methods that update a smaller set of parameters while most pretrained weights remain frozen, reducing optimizer state and adapter checkpoint size.
- **LoRA:** adds trainable low-rank matrices to selected projections. The base weights stay frozen during adapter training; the adapter can later be saved separately or merged.
- **QLoRA:** combines LoRA adapters with a quantized frozen base model, commonly 4-bit, to reduce memory requirements further.

### What this repository does

The Unsloth pipeline loads Qwen3.5-9B with 4-bit loading, attaches PEFT LoRA adapters, and enables training for vision and language layers, attention modules, and MLP modules. It passes `target_modules="all-linear"`, with rank 16, alpha 32, dropout 0.05, and no LoRA bias. A quantized base plus trainable adapters is a QLoRA-style setup. Gradient checkpointing is also enabled to reduce activation memory. The best adapter is merged with the base model after training.

## 7. Training files and setup

### Imports

The primary script imports these central components:

```python
import torch
from datasets import load_dataset
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
```

### Configured values

`config.yaml` is the source of truth for the main settings:

| Setting | Value |
| --- | --- |
| Base model | `models/Qwen3.5-9B` |
| Data | `train_data/vlm_train.jsonl`, `train_data/vlm_val.jsonl` |
| Output | `outputs/qwen35-floorplan-lora` |
| 4-bit loading | Enabled |
| Per-device train/eval batch | 4 / 4 |
| Gradient accumulation | 3 |
| Learning rate | `1e-4` |
| Epoch limit | 10 |
| Image longest side | 1024 pixels |
| Maximum sequence length | 1024 |
| LoRA | rank 16, alpha 32, dropout 0.05 |
| LoRA targets | All linear layers via Unsloth's `target_modules="all-linear"` |
| Save/evaluation interval | Every 1,000 steps |
| Best model metric | `eval_loss` (lower is better) |
| Logging | TensorBoard and `training.log` |

The trainer reads `target_modules: all-linear` from the YAML and passes it to Unsloth.

### Simplified model and trainer setup

The following code blocks document the main stages from the repository's training script. Configuration values are repeated in code so this README can be read as a walkthrough; `train.py` remains the source of truth.

#### 1. Imports and project configuration

```python
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image
from peft import PeftModel
from transformers import AutoModelForImageTextToText, EarlyStoppingCallback
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel

MODEL_DIR = Path("models/Qwen3.5-9B")
TRAIN_FILE = Path("train_data/vlm_train.jsonl")
VAL_FILE = Path("train_data/vlm_val.jsonl")
OUTPUT_DIR = Path("outputs/qwen35-floorplan-lora")
IMAGE_CACHE_DIR = Path("train_data/image_cache")

TRAIN_PERCENTAGE = 100
VALIDATION_PERCENTAGE = 20
NUM_TRAIN_EPOCHS = 10
PER_DEVICE_TRAIN_BATCH_SIZE = 4
PER_DEVICE_EVAL_BATCH_SIZE = 4
GRADIENT_ACCUMULATION_STEPS = 3
LEARNING_RATE = 1e-4
MAX_IMAGE_SIDE = 1024
MAX_SEQ_LENGTH = 1024
LOAD_IN_4BIT = True
SEED = 42
```

#### 2. Load and select JSONL data

Hugging Face Datasets reads one JSON record per line. The script shuffles with a fixed seed before selecting the requested percentage.

```python
def load_jsonl_dataset(train_file: Path, val_file: Path):
    return load_dataset(
        "json",
        data_files={
            "train": str(train_file),
            "validation": str(val_file),
        },
    )


def limit_dataset_by_percentage(dataset_split, percentage: float):
    sample_count = max(1, int(len(dataset_split) * percentage / 100.0))
    return dataset_split.shuffle(seed=SEED).select(range(sample_count))


dataset = load_jsonl_dataset(TRAIN_FILE, VAL_FILE)
train_dataset = limit_dataset_by_percentage(dataset["train"], TRAIN_PERCENTAGE)
val_dataset = limit_dataset_by_percentage(dataset["validation"], VALIDATION_PERCENTAGE)
```

#### 3. Load the model and attach LoRA adapters

Load the model and processor before preprocessing, because the processor supplies the model's chat template. The actual script also checks and fixes EOS/padding token metadata before trainer setup.

```python
model, processor = FastVisionModel.from_pretrained(
    model_name=str(MODEL_DIR),
    load_in_4bit=LOAD_IN_4BIT,
)

model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers=True,
    finetune_language_layers=True,
    finetune_attention_modules=True,
    finetune_mlp_modules=True,
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    target_modules="all-linear",
)

FastVisionModel.for_training(model)
model.config.use_cache = False
```

#### 4. Resize/cache images and format chat messages

Images are converted to RGB, resized only when their longest side exceeds the configured limit, and cached to avoid repeating image decoding/resizing on later runs. The processor formats the messages using the model's chat template.

```python
def resize_and_cache_image(image_path: str, cache_dir: Path, max_image_side: int) -> str:
    cache_dir.mkdir(parents=True, exist_ok=True)
    source = Path(image_path)
    suffix = source.suffix.lower() or ".jpg"
    cache_key = hashlib.md5(str(source.resolve()).encode("utf-8")).hexdigest()
    cached_path = cache_dir / f"{cache_key}{suffix}"

    if cached_path.exists():
        return str(cached_path)

    image = Image.open(source).convert("RGB")
    width, height = image.size
    longest_side = max(width, height)
    if longest_side > max_image_side:
        scale = max_image_side / longest_side
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.BICUBIC,
        )

    image.save(cached_path, quality=95)
    return str(cached_path)


def preprocess_example(example, processor, cache_dir: Path):
    formatted_messages = []
    image_paths = []
    for message in example["messages"]:
        content = []
        for item in message["content"]:
            if item["type"] == "image":
                image_paths.append(
                    resize_and_cache_image(item["image"], cache_dir, MAX_IMAGE_SIDE)
                )
                content.append({"type": "image"})
            elif item["type"] == "text":
                content.append({"type": "text", "text": item["text"]})
        formatted_messages.append({"role": message["role"], "content": content})

    prompt = processor.apply_chat_template(
        formatted_messages,
        tokenize=False,
        add_generation_prompt=False,
    ).strip()
    return {"text": prompt, "image_paths": image_paths}
```

Apply the preprocessing function to both splits (the full script also sets separate train and validation cache directories):

```python
train_dataset = train_dataset.map(
    lambda row: preprocess_example(row, processor, IMAGE_CACHE_DIR / "train"),
    remove_columns=train_dataset.column_names,
    desc="Preprocessing train",
)
val_dataset = val_dataset.map(
    lambda row: preprocess_example(row, processor, IMAGE_CACHE_DIR / "validation"),
    remove_columns=val_dataset.column_names,
    desc="Preprocessing validation",
)
```

#### 5. Collate multimodal batches and mask non-target tokens

The project collator opens each cached image, calls the processor for the text/image batch, then masks padding and image placeholder tokens from the language-model loss.

```python
def build_collate_fn(processor, max_seq_length: int):
    pad_token_id = processor.tokenizer.pad_token_id
    image_token_ids = []
    for name in ("boi_token", "image_token"):
        token = processor.tokenizer.special_tokens_map.get(name)
        if token is not None:
            image_token_ids.append(processor.tokenizer.convert_tokens_to_ids(token))

    def collate_fn(examples):
        texts = [example["text"] for example in examples]
        images = [
            [Image.open(path).convert("RGB") for path in example["image_paths"]]
            for example in examples
        ]
        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == pad_token_id] = -100
        for token_id in image_token_ids:
            labels[labels == token_id] = -100
        batch["labels"] = labels
        return batch

    return collate_fn


collate_fn = build_collate_fn(processor, max_seq_length=MAX_SEQ_LENGTH)
```

#### 6. Configure supervised fine-tuning

```python
training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=NUM_TRAIN_EPOCHS,
    per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
    gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
    learning_rate=LEARNING_RATE,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    save_strategy="steps",
    save_steps=1000,
    eval_strategy="steps",
    eval_steps=1000,
    logging_steps=10,
    bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
    fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
    remove_unused_columns=False,
    dataset_kwargs={"skip_prepare_dataset": True},
    report_to="tensorboard",
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    ddp_find_unused_parameters=False,
    dataloader_num_workers=8,
    dataloader_pin_memory=True,
    dataloader_persistent_workers=True,
    eos_token=processor.eos_token,
    pad_token=processor.tokenizer.pad_token,
)
```

#### 7. Train, evaluate, and resume

```python
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=collate_fn,
    processing_class=processor,
    callbacks=[
        EarlyStoppingCallback(
            early_stopping_patience=15,
            early_stopping_threshold=1e-4,
        ),
        SaveAliasCallback(),  # project callback updates best/latest aliases
    ],
)

# Fresh run:
trainer.train()

# To resume instead, point this at an existing checkpoint:
# trainer.train(resume_from_checkpoint="outputs/qwen35-floorplan-lora/checkpoint-3000")
```

The full script automatically starts `accelerate` across the GPU IDs in `GPU_IDS` when it is run directly. To launch explicitly, use a command such as:

```bash
CUDA_VISIBLE_DEVICES=0,1,2 accelerate launch --multi_gpu --num_processes 3 train.py
```

#### 8. Save the best adapter and merge it into the base model

After training, the main process selects the best adapter checkpoint, loads the base model, applies the adapter, and merges the weights for simpler inference deployment.

```python
best_checkpoint = OUTPUT_DIR / "best-checkpoint"
merged_dir = OUTPUT_DIR / "merged-best"

base_model = AutoModelForImageTextToText.from_pretrained(
    MODEL_DIR,
    torch_dtype=(
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        else torch.float16
    ),
)
if torch.cuda.is_available():
    base_model = base_model.to("cuda")

adapted_model = PeftModel.from_pretrained(base_model, best_checkpoint)
merged_model = adapted_model.merge_and_unload()
merged_model.save_pretrained(merged_dir)
processor.save_pretrained(merged_dir)
```

The complete script includes distributed-process handling, log mirroring, special-token checks, checkpoint alias management, and a `run_summary.json` writer. Refer to `train.py` for those production details.

#### 9. Run and monitor training

```bash
bash finetune_run.sh
bash tensorboard.sh
```

The short excerpts above explain all main stages, but are not a standalone replacement for the repository script: some names, callbacks, and runtime safeguards are defined there.

## 8. Training log

The main script writes `outputs/qwen35-floorplan-lora/training.log` and TensorBoard events to `outputs/tensorboard/qwen35-floorplan-lora/`. It periodically evaluates and saves checkpoints, uses early stopping, and selects the best checkpoint by validation loss.

The checked-in `run_summary.json` records:

- Global step: 18,000
- Training loss: 0.1924
- Best validation loss: 0.4157557
- Best checkpoint: `outputs/qwen35-floorplan-lora/checkpoint-3000`
- Merged model: `outputs/qwen35-floorplan-lora/merged-best`

The final evaluation shown in the log reports `eval_loss = 0.8158` and `eval_mean_token_accuracy = 0.8437` at epoch 0.6247. The summary's best validation loss (0.4157557) is from checkpoint 3000, so these are different evaluation points. Loss and token accuracy are training diagnostics; include held-out examples to show counting, detection, polygon, and generalization quality.

View TensorBoard with:

```bash
tensorboard --logdir outputs/tensorboard/qwen35-floorplan-lora
```

## 9. Output and inference

Artifacts are under `outputs/qwen35-floorplan-lora/`:

- `checkpoint-*/` — intermediate adapter checkpoints and trainer state.
- `best-checkpoint/` — stable alias for the selected adapter.
- `merged-best/` — merged base model and adapter for inference.
- `training.log`, `run_summary.json` — recorded run details.

A minimal loading outline for the merged model:

```python
from transformers import AutoModelForImageTextToText, AutoProcessor

model_path = "outputs/qwen35-floorplan-lora/merged-best"
processor = AutoProcessor.from_pretrained(model_path)
model = AutoModelForImageTextToText.from_pretrained(model_path, device_map="auto")
# Build a user message with image and question using the model chat template,
# process it with processor, then call model.generate(...).
```

Use the project's inference script/notebook for the full image and chat-template flow. Evaluate on held-out floor plans and show sample predictions alongside aggregate measures.

