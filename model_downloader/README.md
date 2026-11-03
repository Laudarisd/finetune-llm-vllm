# Model downloader

`download.py` reads the default Hugging Face repository and local path from the root `config.yaml` and downloads a snapshot using `huggingface_hub`.

```bash
python model_downloader/download.py
```

Override the defaults to download another compatible model:

```bash
python model_downloader/download.py \
  --model-id Qwen/Qwen3-VL-32B-Instruct \
  --destination models_32B
```

For gated repositories, authenticate with `hf auth login` before downloading. Model weights can be large; review licensing/access requirements and make sure the destination has enough free space. Downloaded weights are ignored by Git.
