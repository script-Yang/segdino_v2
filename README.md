# SegDINO V2

Clone the DINOv3 repository:

```bash
git clone https://github.com/facebookresearch/dinov3.git
```

V2 introduces multi-scale feature processing for segmentation.

## Config

All experiment settings live in `config.yaml`.

- Edit dataset paths, checkpoints, and model defaults there.
- Relative paths are resolved from `new_project/`.
- `train.py` reads `SEGDINO_EXPERIMENT` and defaults to `kvasir_train`.
- `test.py` reads `SEGDINO_EXPERIMENT` and defaults to `isic_test`.

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
