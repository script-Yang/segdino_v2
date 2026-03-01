CUDA_VISIBLE_DEVICES=3 python train.py \
  --data_dir /home/sicheng/tmi_re/segdinov3/segdata/kvasir \
  --img_size 256 \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --repo_dir ./dinov3 \
  --dino_size s \
  --dino_ckpt /vip_media/sicheng/DataShare/tmi_re/segdino_good/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

CUDA_VISIBLE_DEVICES=0 python test.py \
  --data_dir /home/sicheng/tmi_re/segdinov3/segdata/kvasir \
  --batch_size 1 \
  --img_size 256 \
  --repo_dir ./dinov3 \
  --dino_size s \
  --dino_ckpt /vip_media/sicheng/DataShare/tmi_re/segdino_good/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth \
  --ckpt /home/sicheng/tmi_re/segdino_miccai/checkpoints/best_ep46_dice_0.8911_iou_0.8267.pth

CUDA_VISIBLE_DEVICES=1 python train.py \
  --data_dir /home/sicheng/tmi_re/segdinov3/segdata/tn3k \
  --img_size 256 \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --repo_dir ./dinov3 \
  --dino_size s \
  --dino_ckpt /vip_media/sicheng/DataShare/tmi_re/segdino_good/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth

CUDA_VISIBLE_DEVICES=0 python train.py \
  --data_dir /home/sicheng/tmi_re/segdinov3/segdata/ISIC \
  --img_size 256 \
  --batch_size 4 \
  --epochs 50 \
  --lr 1e-4 \
  --repo_dir ./dinov3 \
  --dino_size s \
  --dino_ckpt /vip_media/sicheng/DataShare/tmi_re/segdino_good/web_pth/dinov3_vits16_pretrain_lvd1689m-08c60483.pth