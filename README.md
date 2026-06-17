# SegDINO: Introducing Multi-Scale Structure into DINO for Efficient Medical Image Segmentation

<h5 align="center">

[![arXiv](https://img.shields.io/badge/Arxiv-2511.06863-AD1C18.svg?logo=arXiv)](https://arxiv.org/abs/2606.17972)

</h5>

## News

> **2026.06** Our [paper](https://arxiv.org/abs/2606.17972) was released on arXiv.  
> **2026.05** SegDINO was selected for early acceptance at MICCAI 2026.  
> **2026.06** Code released.  
> **2025.09** Check out SegDINO-V1 [here](https://github.com/script-Yang/segdino).

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

## Citation
```bibtex
@article{yang2026segdino,
  title={SegDINO: Introducing Multi-Scale Structure into DINO for Efficient Medical Image Segmentation},
  author={Yang, Sicheng and Wang, Hongqiu and Xing, Zhaohu and Chen, Sixiang and Yang, Qiuxia and Mao, Yize and Yang, Guang and Zhu, Lei},
  journal={arXiv preprint arXiv:2606.17972},
  year={2026}
}
```