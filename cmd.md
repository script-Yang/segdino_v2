Use `config.yaml` plus the shell scripts in `scripts/`.

Examples:

bash scripts/train_kvasir.sh
bash scripts/train_tn3k.sh
bash scripts/train_isic.sh
bash scripts/test_kvasir.sh
bash scripts/test_isic.sh

Override the GPU when needed:

CUDA_VISIBLE_DEVICES=1 bash scripts/train_kvasir.sh
