# segdino_v2

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_dir ./segdata/kvasir \
  --img_size 256 \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --repo_dir ./dinov3 \
  --dino_size s \
  --dino_ckpt ./dinov3_vits16_pretrain_lvd1689m-08c60483.pth
```

```bash
CUDA_VISIBLE_DEVICES=1 python test.py \
  --data_dir ./segdata/kvasir \
  --batch_size 1 \
  --img_size 256 \
  --repo_dir ./dinov3 \
  --dino_size s \
  --dino_ckpt ./dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --ckpt xxx.pth
```