import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.distributed as dist
from datasets import load_dataset
from PIL import Image
from peft import PeftModel 
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
    EarlyStoppingCallback,
    TrainerCallback,
)
from trl import SFTConfig, SFTTrainer
from unsloth import FastVisionModel
import yaml


# ============================================================
# Configuration
# ============================================================
# Keep tunable settings in config.yaml. Relative paths are anchored at this file,
# so the project can be launched from any working directory.
PROJECT_ROOT = Path(__file__).resolve().parent
CONFIG_FILE = Path(os.environ.get("CONFIG_FILE", PROJECT_ROOT / "config.yaml"))
if not CONFIG_FILE.is_absolute():
    CONFIG_FILE = PROJECT_ROOT / CONFIG_FILE
with CONFIG_FILE.open("r", encoding="utf-8") as config_stream:
    CONFIG = yaml.safe_load(config_stream)


def configured_path(environment_name: str, config_value: str) -> Path:
    """Resolve a config path, allowing an environment variable override."""
    value = Path(os.environ.get(environment_name, config_value))
    return value if value.is_absolute() else PROJECT_ROOT / value


MODEL_DIR = configured_path("MODEL_DIR", CONFIG["model"]["directory"])
TRAIN_FILE = configured_path("TRAIN_FILE", CONFIG["data"]["train_file"])
VAL_FILE = configured_path("VAL_FILE", CONFIG["data"]["validation_file"])
OUTPUT_DIR = configured_path("OUTPUT_DIR", CONFIG["training"]["output_directory"])
MERGED_DIR = configured_path("MERGED_DIR", CONFIG["model"]["merged_directory"])
TENSORBOARD_LOGGING_DIR = configured_path(
    "TENSORBOARD_LOGGING_DIR", CONFIG["training"]["tensorboard_directory"]
)
IMAGE_CACHE_DIR = configured_path(
    "IMAGE_CACHE_DIR", CONFIG["data"]["image_cache_directory"]
)
TRAINING_LOG_FILE = "training.log"

# Data controls: use deterministic percentages for quick experiments or full runs.
TRAIN_PERCENTAGE = float(CONFIG["data"]["train_percentage"])
VALIDATION_PERCENTAGE = float(CONFIG["data"]["validation_percentage"])
MAX_IMAGE_SIDE = int(CONFIG["data"]["max_image_side"])
MAX_SEQ_LENGTH = int(CONFIG["data"]["max_sequence_length"])

# GPU_IDS can override the comma-separated GPU list in config.yaml.
GPU_IDS = os.environ.get("GPU_IDS", str(CONFIG["training"]["gpu_ids"]))
SEED = int(CONFIG["training"]["seed"])
NUM_TRAIN_EPOCHS = int(CONFIG["training"]["epochs"])
PER_DEVICE_TRAIN_BATCH_SIZE = int(CONFIG["training"]["train_batch_size_per_device"])
PER_DEVICE_EVAL_BATCH_SIZE = int(CONFIG["training"]["evaluation_batch_size_per_device"])
GRADIENT_ACCUMULATION_STEPS = int(CONFIG["training"]["gradient_accumulation_steps"])
LEARNING_RATE = float(CONFIG["training"]["learning_rate"])
LEARNING_SCHEDULER_TYPE = str(CONFIG["training"]["scheduler"])
WARMUP_STEPS = int(CONFIG["training"]["warmup_steps"])
GRADIENT_CHECKPOINTING = bool(CONFIG["training"]["gradient_checkpointing"])
LOGGING_STEPS = int(CONFIG["training"]["logging_steps"])
SAVE_STEPS = int(CONFIG["training"]["save_steps"])
EVAL_STEPS = int(CONFIG["training"]["evaluation_steps"])
REPORT_TO = CONFIG["training"]["report_to"]
RESUME_FROM_CHECKPOINT = CONFIG["training"].get("resume_from_checkpoint")

# LoRA/QLoRA controls. The base model is loaded at 4-bit when enabled.
LOAD_IN_4BIT = bool(CONFIG["model"]["load_in_4bit"])
LORA_R = int(CONFIG["lora"]["rank"])
LORA_ALPHA = int(CONFIG["lora"]["alpha"])
LORA_DROPOUT = float(CONFIG["lora"]["dropout"])
LORA_BIAS = str(CONFIG["lora"]["bias"])
LORA_TARGET_MODULES = CONFIG["lora"]["target_modules"]
LORA_ENSURE_WEIGHT_TYING = False

# Validation controls: lower eval_loss is better by default.
EARLY_STOPPING_PATIENCE = int(CONFIG["evaluation"]["early_stopping_patience"])
EARLY_STOPPING_THRESHOLD = float(CONFIG["evaluation"]["early_stopping_threshold"])
BEST_MODEL_METRIC = str(CONFIG["evaluation"]["best_metric"])
BEST_MODEL_GREATER_IS_BETTER = bool(CONFIG["evaluation"]["greater_is_better"])

# Dataloader settings primarily affect CPU memory and image loading throughput.
DATALOADER_NUM_WORKERS = int(CONFIG["dataloader"]["workers"])
DATALOADER_PIN_MEMORY = bool(CONFIG["dataloader"]["pin_memory"])
DATALOADER_PERSISTENT_WORKERS = bool(CONFIG["dataloader"]["persistent_workers"])

# Qwen chat EOS used when local tokenizer metadata is incomplete.
DEFAULT_QWEN_EOS_TOKEN = "<|im_end|>"


class TeeStream:
    """Write stream output to terminal and log file."""

    def __init__(self, terminal_stream, log_stream):
        self.terminal_stream = terminal_stream
        self.log_stream = log_stream

    def write(self, data):
        self.terminal_stream.write(data)
        self.log_stream.write(data)
        return len(data)

    def flush(self):
        self.terminal_stream.flush()
        self.log_stream.flush()

    def isatty(self):
        return self.terminal_stream.isatty()


def setup_training_log_file(output_dir: Path) -> Path | None:
    """
    Mirror stdout/stderr to a log file.

    Enabled only on main process to avoid interleaved multi-rank logs.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if local_rank != 0:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / TRAINING_LOG_FILE
    log_file = open(log_path, "a", encoding="utf-8")
    log_file.write(
        f"\n\n===== Training run started at {datetime.now(timezone.utc).isoformat()} UTC =====\n"
    )
    log_file.flush()

    sys.stdout = TeeStream(sys.__stdout__, log_file)
    sys.stderr = TeeStream(sys.__stderr__, log_file)
    return log_path


def maybe_launch_with_accelerate():
    """
    If the script is started normally, relaunch it with `accelerate launch`
    across multiple GPUs.

    This function avoids relaunching if:
    - LOCAL_RANK already exists (meaning accelerate already launched it), or
    - RUN_WITH_ACCELERATE is already set.
    """
    if os.environ.get("LOCAL_RANK") is not None:
        return
    if os.environ.get("RUN_WITH_ACCELERATE") == "1":
        return

    gpu_ids = [gpu_id.strip() for gpu_id in GPU_IDS.split(",") if gpu_id.strip()]
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(gpu_ids)
    env["TOKENIZERS_PARALLELISM"] = "false"
    env["RUN_WITH_ACCELERATE"] = "1"

    command = [
        "accelerate",
        "launch",
        "--multi_gpu",
        "--num_processes",
        str(len(gpu_ids)),
        Path(__file__).name,
    ]

    print(f"Launching distributed training on GPUs: {env['CUDA_VISIBLE_DEVICES']}")
    os.execvpe(command[0], command, env)


def load_jsonl_dataset(train_file: Path, val_file: Path):
    """
    Load training and validation JSONL files using Hugging Face datasets.
    """
    data_files = {
        "train": str(train_file),
        "validation": str(val_file),
    }
    return load_dataset("json", data_files=data_files)


def limit_dataset_by_percentage(dataset_split, percentage: float):
    """
    Shuffle the dataset and keep only the requested percentage.
    """
    total = len(dataset_split)
    limit = max(1, int(total * (percentage / 100.0)))
    shuffled_dataset = dataset_split.shuffle(seed=SEED)
    return shuffled_dataset.select(range(limit))


def ensure_special_tokens_map(model_dir: Path) -> dict[str, str]:
    """
    Ensure the local model directory has a special_tokens_map.json file.

    Some Unsloth tokenizer paths behave better when this file is present for
    local checkpoints. We source the values from tokenizer_config.json.
    """
    tokenizer_config_path = model_dir / "tokenizer_config.json"
    special_tokens_map_path = model_dir / "special_tokens_map.json"

    with tokenizer_config_path.open("r", encoding="utf-8") as handle:
        tokenizer_config = json.load(handle)

    special_tokens_map = {
        "eos_token": tokenizer_config.get("eos_token", DEFAULT_QWEN_EOS_TOKEN),
    }
    pad_token = tokenizer_config.get("pad_token")
    if pad_token is not None:
        special_tokens_map["pad_token"] = pad_token

    if not special_tokens_map_path.exists():
        with special_tokens_map_path.open("w", encoding="utf-8") as handle:
            json.dump(special_tokens_map, handle, indent=2)
            handle.write("\n")

    return special_tokens_map


def prepare_unsloth_model(model_dir: Path, load_in_4bit: bool):
    """
    Load the multimodal model and processor through Unsloth, then attach LoRA.
    """
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    special_tokens_map = ensure_special_tokens_map(model_dir)
    configured_eos_token = special_tokens_map.get("eos_token", DEFAULT_QWEN_EOS_TOKEN)

    model, processor = FastVisionModel.from_pretrained(
        model_name=str(model_dir),
        load_in_4bit=load_in_4bit,
        device_map={"": local_rank} if torch.cuda.is_available() else None,
    )

    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "right"
        tokenizer = processor.tokenizer

        # TRL reads special tokens from the processing class during trainer setup.
        # Qwen's processor may leave these on the tokenizer only, so mirror them
        # onto the processor explicitly to avoid placeholder EOS fallbacks.
        if tokenizer.eos_token in (None, "eos_token", "<EOS_TOKEN>"):
            tokenizer.eos_token = configured_eos_token
        if tokenizer.eos_token_id is None:
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)

        if tokenizer.eos_token_id is None and configured_eos_token != tokenizer.eos_token:
            tokenizer.eos_token = configured_eos_token
            tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids(configured_eos_token)

        if tokenizer.eos_token is None:
            raise ValueError("Loaded tokenizer does not define an eos_token.")
        if tokenizer.eos_token_id is None:
            raise ValueError(
                f"Loaded tokenizer eos_token {tokenizer.eos_token!r} is not in the vocabulary."
            )

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
            tokenizer.pad_token_id = tokenizer.eos_token_id

        processor.eos_token = tokenizer.eos_token
        processor.eos_token_id = tokenizer.eos_token_id
        processor.pad_token = tokenizer.pad_token
        processor.pad_token_id = tokenizer.pad_token_id

        print(
            "Processor special tokens:"
            f" eos_token={processor.eos_token!r}"
            f" eos_token_id={processor.eos_token_id}"
            f" pad_token={processor.pad_token!r}"
            f" pad_token_id={processor.pad_token_id}"
        )

    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=bool(CONFIG["lora"]["finetune_vision_layers"]),
        finetune_language_layers=bool(CONFIG["lora"]["finetune_language_layers"]),
        finetune_attention_modules=bool(CONFIG["lora"]["finetune_attention_modules"]),
        finetune_mlp_modules=bool(CONFIG["lora"]["finetune_mlp_modules"]),
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias=LORA_BIAS,
        target_modules=LORA_TARGET_MODULES,
    )

    FastVisionModel.for_training(model)
    return model, processor


def replace_dir(source_dir: Path, target_dir: Path):
    """
    Replace target directory with source directory contents.
    Used for creating stable aliases like:
    - best-checkpoint
    - latest-checkpoint
    """
    if not source_dir.exists():
        return
    if target_dir.exists() or target_dir.is_symlink():
        shutil.rmtree(target_dir, ignore_errors=True)
    shutil.copytree(source_dir, target_dir)


def save_checkpoint_aliases(trainer, output_dir: Path):
    """
    Create/update helper folder:
    - best-checkpoint
    """
    best_dir = output_dir / "best-checkpoint"
    latest_dir = output_dir / "latest-checkpoint"

    if latest_dir.exists() or latest_dir.is_symlink():
        shutil.rmtree(latest_dir, ignore_errors=True)

    if trainer.state.best_model_checkpoint:
        replace_dir(Path(trainer.state.best_model_checkpoint), best_dir)


class SaveAliasCallback(TrainerCallback):
    """
    Callback that keeps the best-checkpoint alias updated during training.
    """

    def on_save(self, args, state, control, **kwargs):
        return control

    def on_evaluate(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return control

        if state.best_model_checkpoint:
            replace_dir(
                Path(state.best_model_checkpoint),
                Path(args.output_dir) / "best-checkpoint",
            )

        return control


def make_cached_image_path(image_path: str, cache_dir: Path) -> Path:
    """
    Generate a deterministic cache file path for an image.
    """
    src = Path(image_path)
    suffix = src.suffix.lower() if src.suffix else ".jpg"
    key = hashlib.md5(str(src.resolve()).encode("utf-8")).hexdigest()
    return cache_dir / f"{key}{suffix}"


def resize_and_cache_image(image_path: str, cache_dir: Path, max_image_side: int) -> str:
    """
    Resize image if needed and store a cached copy.
    This reduces repeated image decoding/resizing overhead.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path(image_path)
    if not source_path.is_absolute():
        source_path = PROJECT_ROOT / source_path
    cached_path = make_cached_image_path(str(source_path), cache_dir)

    if cached_path.exists():
        return str(cached_path)

    image = Image.open(source_path).convert("RGB")
    width, height = image.size
    longest_side = max(width, height)

    if longest_side > max_image_side:
        scale = max_image_side / longest_side
        resized_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        image = image.resize(resized_size, Image.Resampling.BICUBIC)

    save_format = "PNG" if cached_path.suffix.lower() == ".png" else "JPEG"
    image.save(cached_path, format=save_format, quality=95)
    return str(cached_path)


def preprocess_example(example, processor, image_cache_dir: Path, max_image_side: int):
    """
    Convert one chat-style multimodal example into:
    - formatted text prompt
    - cached image path list
    """
    messages_for_template = []
    cached_image_paths = []

    for message in example["messages"]:
        new_content = []

        for item in message["content"]:
            if item["type"] == "image":
                cached_path = resize_and_cache_image(
                    image_path=item["image"],
                    cache_dir=image_cache_dir,
                    max_image_side=max_image_side,
                )
                cached_image_paths.append(cached_path)
                new_content.append({"type": "image"})

            elif item["type"] == "text":
                new_content.append({"type": "text", "text": item["text"]})

        messages_for_template.append(
            {
                "role": message["role"],
                "content": new_content,
            }
        )

    text = processor.apply_chat_template(
        messages_for_template,
        tokenize=False,
        add_generation_prompt=False,
    ).strip()

    return {
        "text": text,
        "image_paths": cached_image_paths,
    }


def preprocess_dataset(dataset_split, processor, image_cache_dir: Path, max_image_side: int, split_name: str):
    """
    Preprocess the whole dataset split.
    """
    print(f"Preprocessing {split_name} dataset...")
    return dataset_split.map(
        lambda example: preprocess_example(
            example=example,
            processor=processor,
            image_cache_dir=image_cache_dir,
            max_image_side=max_image_side,
        ),
        remove_columns=dataset_split.column_names,
        desc=f"Preprocessing {split_name}",
    )


def build_collate_fn(processor, max_seq_length: int):
    """
    Build a batch collation function for multimodal examples.

    Important:
    - pad tokens are masked with -100 in labels
    - image placeholder tokens are also masked with -100
    """
    pad_token_id = processor.tokenizer.pad_token_id

    image_token_ids = []
    for token_name in ("boi_token", "image_token"):
        token_value = processor.tokenizer.special_tokens_map.get(token_name)
        if token_value is not None:
            image_token_ids.append(
                processor.tokenizer.convert_tokens_to_ids(token_value)
            )

    def collate_fn(examples):
        texts = [example["text"] for example in examples]

        images = []
        for example in examples:
            example_images = [
                Image.open(image_path).convert("RGB")
                for image_path in example["image_paths"]
            ]
            images.append(example_images)

        batch = processor(
            text=texts,
            images=images,
            padding=True,
            truncation=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()

        # Ignore pad tokens in loss.
        labels[labels == pad_token_id] = -100

        # Ignore image special tokens in loss.
        for token_id in image_token_ids:
            labels[labels == token_id] = -100

        batch["labels"] = labels
        return batch

    return collate_fn


def cleanup_distributed():
    """
    Clean up NCCL / torch.distributed state to avoid the warning:
    'process group has NOT been destroyed'
    """
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def main():
    maybe_launch_with_accelerate()
    log_path = setup_training_log_file(OUTPUT_DIR)
    if log_path:
        print(f"Writing training logs to: {log_path}")

    if not MODEL_DIR.exists():
        raise FileNotFoundError(f"Model directory not found: {MODEL_DIR}")
    if not TRAIN_FILE.exists():
        raise FileNotFoundError(f"Train file not found: {TRAIN_FILE}")
    if not VAL_FILE.exists():
        raise FileNotFoundError(f"Validation file not found: {VAL_FILE}")

    print(f"Loading model from: {MODEL_DIR}")
    model, processor = prepare_unsloth_model(MODEL_DIR, LOAD_IN_4BIT)
    model.config.use_cache = False

    print("Loading datasets...")
    dataset = load_jsonl_dataset(TRAIN_FILE, VAL_FILE)

    train_dataset = limit_dataset_by_percentage(dataset["train"], TRAIN_PERCENTAGE)
    val_dataset = limit_dataset_by_percentage(dataset["validation"], VALIDATION_PERCENTAGE)

    print(f"Train samples before preprocessing: {len(train_dataset)}")
    print(f"Validation samples before preprocessing: {len(val_dataset)}")
    print(f"Train percentage: {TRAIN_PERCENTAGE}%")
    print(f"Validation percentage: {VALIDATION_PERCENTAGE}%")

    train_cache_dir = IMAGE_CACHE_DIR / "train"
    val_cache_dir = IMAGE_CACHE_DIR / "validation"

    train_dataset = preprocess_dataset(
        dataset_split=train_dataset,
        processor=processor,
        image_cache_dir=train_cache_dir,
        max_image_side=MAX_IMAGE_SIDE,
        split_name="train",
    )

    val_dataset = preprocess_dataset(
        dataset_split=val_dataset,
        processor=processor,
        image_cache_dir=val_cache_dir,
        max_image_side=MAX_IMAGE_SIDE,
        split_name="validation",
    )

    print(f"Train samples after preprocessing: {len(train_dataset)}")
    print(f"Validation samples after preprocessing: {len(val_dataset)}")

    # collate_fn = UnslothVisionDataCollator(
    #     model,
    #     processor,
    #     train_on_responses_only=True,
    # )
    collate_fn = build_collate_fn(
        processor=processor,
        max_seq_length=MAX_SEQ_LENGTH,
    )

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_TRAIN_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_TRAIN_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type=LEARNING_SCHEDULER_TYPE,
        #warmup_ratio=WARMUP_RATIO,  # ~3% of steps as warmup
        warmup_steps=WARMUP_STEPS,
        gradient_checkpointing=GRADIENT_CHECKPOINTING,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        eval_strategy="steps",
        eval_steps=EVAL_STEPS,
        logging_steps=LOGGING_STEPS,
        seed=SEED,
        bf16=torch.cuda.is_available() and torch.cuda.is_bf16_supported(),
        fp16=torch.cuda.is_available() and not torch.cuda.is_bf16_supported(),
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        report_to=REPORT_TO,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model=BEST_MODEL_METRIC,
        greater_is_better=BEST_MODEL_GREATER_IS_BETTER,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=DATALOADER_NUM_WORKERS,
        dataloader_pin_memory=DATALOADER_PIN_MEMORY,
        dataloader_persistent_workers=DATALOADER_PERSISTENT_WORKERS,
        eos_token=processor.eos_token,
        pad_token=processor.tokenizer.pad_token,
    )
    if training_args.eos_token in (None, "eos_token", "<EOS_TOKEN>"):
        training_args.eos_token = processor.eos_token
    if training_args.pad_token is None:
        training_args.pad_token = processor.pad_token

    resolved_eos_token_id = processor.tokenizer.convert_tokens_to_ids(training_args.eos_token)
    if resolved_eos_token_id is None:
        raise ValueError(
            f"Resolved EOS token {training_args.eos_token!r} is still not in the tokenizer vocabulary."
        )
    print(
        "Trainer special tokens:"
        f" eos_token={training_args.eos_token!r}"
        f" eos_token_id={resolved_eos_token_id}"
        f" pad_token={training_args.pad_token!r}"
    )

    # Newer versions may prefer environment variable instead of logging_dir.
    os.environ["TENSORBOARD_LOGGING_DIR"] = str(TENSORBOARD_LOGGING_DIR)

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
        processing_class=processor,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                early_stopping_threshold=EARLY_STOPPING_THRESHOLD,
            ),
            SaveAliasCallback(),
        ],
    )

    # Continue from checkpoint when configured, otherwise start from scratch.
    if RESUME_FROM_CHECKPOINT:
        resume_path = str(OUTPUT_DIR / RESUME_FROM_CHECKPOINT)
        print(f"Resuming from checkpoint: {resume_path}")
        trainer.train(resume_from_checkpoint=resume_path)
    else:
        print("Starting a fresh training run.")
        trainer.train()


    if trainer.is_world_process_zero():
        save_checkpoint_aliases(trainer, OUTPUT_DIR)
        print(f"Saved best-checkpoint alias in: {OUTPUT_DIR}")
        
        best_ckpt = OUTPUT_DIR / "best-checkpoint"
        merged_dir = MERGED_DIR
        print(f"Merging adapter from: {best_ckpt}")

        merge_dtype = (
            torch.bfloat16
            if torch.cuda.is_available() and torch.cuda.is_bf16_supported()
            else torch.float16
        )

        base_for_merge = AutoModelForImageTextToText.from_pretrained(
            MODEL_DIR,
            torch_dtype=merge_dtype,
        )

        if torch.cuda.is_available():
            base_for_merge = base_for_merge.to("cuda")

        peft_for_merge = PeftModel.from_pretrained(base_for_merge, best_ckpt)
        merged_model = peft_for_merge.merge_and_unload()

        merged_model.save_pretrained(merged_dir)
        processor.save_pretrained(merged_dir)
        print(f"Saved merged full model to: {merged_dir}")

    summary = {
        "model_dir": str(MODEL_DIR),
        "train_file": str(TRAIN_FILE),
        "val_file": str(VAL_FILE),
        "output_dir": str(OUTPUT_DIR),
        "tensorboard_logging_dir": str(TENSORBOARD_LOGGING_DIR),
        "image_cache_dir": str(IMAGE_CACHE_DIR),
        "train_percentage": TRAIN_PERCENTAGE,
        "validation_percentage": VALIDATION_PERCENTAGE,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "num_train_epochs": NUM_TRAIN_EPOCHS,
        "per_device_train_batch_size": PER_DEVICE_TRAIN_BATCH_SIZE,
        "per_device_eval_batch_size": PER_DEVICE_EVAL_BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "learning_rate": LEARNING_RATE,
        "save_steps": SAVE_STEPS,
        "eval_steps": EVAL_STEPS,
        "max_image_side": MAX_IMAGE_SIDE,
        "max_seq_length": MAX_SEQ_LENGTH,
        "load_in_4bit": LOAD_IN_4BIT,
        "gradient_checkpointing": GRADIENT_CHECKPOINTING,
        "resume_from_checkpoint": (
            str(OUTPUT_DIR / RESUME_FROM_CHECKPOINT) if RESUME_FROM_CHECKPOINT else None
        ),
        "dataloader_num_workers": DATALOADER_NUM_WORKERS,
        "dataloader_pin_memory": DATALOADER_PIN_MEMORY,
        "dataloader_persistent_workers": DATALOADER_PERSISTENT_WORKERS,
        "lora_r": LORA_R,
        "lora_alpha": LORA_ALPHA,
        "lora_dropout": LORA_DROPOUT,
        "lora_target_modules": LORA_TARGET_MODULES,
        "lora_ensure_weight_tying": LORA_ENSURE_WEIGHT_TYING,
        "seed": SEED,
        "best_model_metric": BEST_MODEL_METRIC,
        "greater_is_better": BEST_MODEL_GREATER_IS_BETTER,
        "best_model_checkpoint": trainer.state.best_model_checkpoint,
        "best_metric": trainer.state.best_metric,
        "global_step": trainer.state.global_step,
        "training_log_file": str(OUTPUT_DIR / TRAINING_LOG_FILE),
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "run_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    cleanup_distributed()


if __name__ == "__main__":
    main()
