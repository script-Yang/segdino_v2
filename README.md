# SegDINO V2

SegDINOv2 introduces multi-scale feature processing for segmentation.

Clone the DINOv3 repository:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

## Config

All experiment settings live in `config.yaml`.

- Edit dataset paths, checkpoints, and model defaults there.
- Relative paths are resolved from `new_project/`.
- `decoder_dim` defaults to `256`.
- For more challenging segmentation tasks, you can increase it to `512`.

## Run

Use the scripts in `scripts/`:

```bash
bash scripts/train_kvasir.sh
bash scripts/train_tn3k.sh
bash scripts/train_isic.sh
bash scripts/test_kvasir.sh
bash scripts/test_isic.sh
```

You can override the GPU without touching Python:

```bash
CUDA_VISIBLE_DEVICES=1 bash scripts/train_kvasir.sh
```
