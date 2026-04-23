import os
import csv
import math
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from config_loader import DEFAULT_TEST_EXPERIMENT, EXPERIMENT_ENV_VAR, get_test_config
from mydataset import FolderDataset, TestTransform
from runtime import build_model, get_device, summarize_parameters
from utils import (
    dice_binary_numpy, iou_binary_numpy, hd95_binary_numpy, 
    save_eval_visuals, load_ckpt_flex
)

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
    experiment_name = os.environ.get(EXPERIMENT_ENV_VAR, DEFAULT_TEST_EXPERIMENT)
    config = get_test_config(experiment_name)

    save_root = Path(config.save_root) / f"{config.dataset.name}_segdino_{config.model.dino_size}_test"
    vis_dir = save_root / "vis"
    csv_path = save_root / "metrics.csv"

    device = get_device()
    model, backbone = build_model(config.model, device)
    total_params, backbone_params, other_params = summarize_parameters(model, backbone)

    print(f"Testing experiment: {config.name}")
    print(f"Dataset: {config.dataset.data_dir}")
    print(f"Checkpoint: {config.ckpt_path}")

    print("=" * 60)
    print(f"Model Parameter Counts:")
    print(f"  - Backbone (DINOv3): {backbone_params / 1e6:>8.2f} M")
    print(f"  - Other (Decoder)  : {other_params / 1e6:>8.2f} M")
    print(f"  - Total Parameters : {total_params / 1e6:>8.2f} M")
    print("=" * 60)

    load_ckpt_flex(model, config.ckpt_path, map_location=device)

    test_dataset = FolderDataset(
        root=config.dataset.data_dir,
        split="test",
        transform=TestTransform(img_size=(config.dataset.img_size, config.dataset.img_size))
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    run_test(
        model, test_loader, device, 
        dice_thr=config.dice_thr,
        vis_dir=str(vis_dir),
        csv_path=str(csv_path),
    )

if __name__ == "__main__":
    main()
