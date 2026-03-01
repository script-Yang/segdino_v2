import os
import csv
import math
import numpy as np
import torch
from tqdm import tqdm
from mydataset import FolderDataset, TestTransform
from utils import (
    dice_binary_numpy, iou_binary_numpy, hd95_binary_numpy, 
    save_eval_visuals, load_ckpt_flex
)
from dpt import DPT

@torch.no_grad()
def run_test(model, loader, device, dice_thr=0.5, vis_dir=None, csv_path=None):
    model.eval()
    if vis_dir: os.makedirs(vis_dir, exist_ok=True)

    rows, dices, ious, hd95s = [], [], [], []
    idx_global = 0

    pbar = tqdm(loader)
    for inputs, targets, ids in pbar:
        case_ids = list(ids)
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        probs = torch.sigmoid(logits)
        preds = (probs > dice_thr).float()

        B = inputs.size(0)
        for b in range(B):
            gt = targets[b, 0].detach().cpu().numpy() > 0.5
            pr = preds[b, 0].detach().cpu().numpy() > 0.5

            dsc = dice_binary_numpy(pr, gt)
            iou = iou_binary_numpy(pr, gt)
            hd95 = hd95_binary_numpy(pr, gt)

            dices.append(dsc)
            ious.append(iou)
            hd95s.append(hd95)
            rows.append({"id": case_ids[b], "dice": dsc, "iou": iou, "hd95": hd95})

            if vis_dir is not None:
                save_eval_visuals(idx_global, inputs[b], logits[b], targets[b], vis_dir, thr=dice_thr, fname_prefix="test")
            idx_global += 1

        valid_hd95s = [x for x in hd95s if np.isfinite(x)]
        pbar.set_postfix(
            mDice=f"{np.mean(dices):.4f}", 
            mIoU=f"{np.mean(ious):.4f}", 
            mHD95=f"{np.mean(valid_hd95s) if valid_hd95s else 0.0:.2f}"
        )

    mean_dice = float(np.mean(dices)) if dices else 0.0
    mean_iou  = float(np.mean(ious))  if ious  else 0.0
    finite_hd = [x for x in hd95s if np.isfinite(x)]
    mean_hd95 = float(np.mean(finite_hd)) if len(finite_hd) > 0 else float("inf")

    if csv_path is not None:
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["id", "dice", "iou", "hd95"])
            writer.writeheader()
            for r in rows: writer.writerow(r)
            writer.writerow({"id": "MEAN", "dice": mean_dice, "iou": mean_iou, "hd95": mean_hd95})

    print("=" * 60)
    print(f"Dice={mean_dice:.4f}  IoU={mean_iou:.4f}  HD95={mean_hd95 if math.isfinite(mean_hd95) else 'inf'}")
    print("=" * 60)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./kvasir")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--dice_thr", type=float, default=0.5)
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--save_root", type=str, default="./runs_test")
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    args = parser.parse_args()

    dataset_name = os.path.basename(args.data_dir.rstrip("/"))
    save_root = os.path.join(args.save_root, f"{dataset_name}_segdino_{args.dino_size}_test")
    vis_dir   = os.path.join(save_root, "vis")
    csv_path  = os.path.join(save_root, "metrics.csv")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)

    model = DPT(nclass=1, backbone=backbone).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    backbone_params = sum(p.numel() for p in backbone.parameters())
    other_params = total_params - backbone_params

    print("=" * 60)
    print(f"Model Parameter Counts:")
    print(f"  - Backbone (DINOv3): {backbone_params / 1e6:>8.2f} M")
    print(f"  - Other (Decoder)  : {other_params / 1e6:>8.2f} M")
    print(f"  - Total Parameters : {total_params / 1e6:>8.2f} M")
    print("=" * 60)

    load_ckpt_flex(model, args.ckpt, map_location=device)

    test_dataset = FolderDataset(
        root=args.data_dir,
        split="test",
        transform=TestTransform(img_size=(args.img_size, args.img_size))
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )

    run_test(
        model, test_loader, device, 
        dice_thr=args.dice_thr, vis_dir=vis_dir, csv_path=csv_path
    )

if __name__ == "__main__":
    main()