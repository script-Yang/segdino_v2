import os
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from config_loader import DEFAULT_TRAIN_EXPERIMENT, EXPERIMENT_ENV_VAR, get_train_config
from mydataset import FolderDataset, TrainTransform, TestTransform
from runtime import build_model, get_device
from utils import dice_coeff, iou_coeff

@torch.no_grad()
def evaluate(model, test_loader, device):
    model.eval()
    running_dice = 0.0
    running_iou = 0.0
    
    pbar = tqdm(test_loader, leave=False)
    for step, (inputs, targets, _) in enumerate(pbar):
        inputs, targets = inputs.to(device), targets.to(device)
        logits = model(inputs)
        
        dice = dice_coeff(logits, targets)
        iou = iou_coeff(logits, targets)
        
        running_dice += dice.item()
        running_iou += iou.item()
        
        pbar.set_postfix(Val_Dice=f"{running_dice/(step+1):.4f}", Val_IoU=f"{running_iou/(step+1):.4f}")
        
    return running_dice / len(test_loader), running_iou / len(test_loader)


def main():
    experiment_name = os.environ.get(EXPERIMENT_ENV_VAR, DEFAULT_TRAIN_EXPERIMENT)
    config = get_train_config(experiment_name)
    device = get_device()
    save_dir = Path(config.save_dir) / config.dataset.name

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Training experiment: {config.name}")
    print(f"Dataset: {config.dataset.data_dir}")
    print(f"Decoder dim: {config.model.decoder_dim}")

    train_dataset = FolderDataset(
        root=config.dataset.data_dir,
        split="train",
        transform=TrainTransform(img_size=(config.dataset.img_size, config.dataset.img_size))
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.train_workers,
    )

    test_dataset = FolderDataset(
        root=config.dataset.data_dir,
        split="test",
        transform=TestTransform(img_size=(config.dataset.img_size, config.dataset.img_size))
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config.val_workers,
    )

    model, _ = build_model(config.model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_dice = 0.0
    best_ckpt_path = ""

    for epoch in range(config.epochs):
        model.train()
        pbar = tqdm(train_loader)
        
        running_loss = 0.0
        running_dice = 0.0
        
        for step, (inputs, targets, _) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            optimizer.zero_grad()
            logits = model(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

            dice = dice_coeff(logits, targets)
            running_loss += loss.item()
            running_dice += dice.item()

            avg_loss = running_loss / (step + 1)
            avg_dice = running_dice / (step + 1)
            
            pbar.set_postfix(Loss=f"{avg_loss:.4f}", Dice=f"{avg_dice:.4f}")

        val_dice, val_iou = evaluate(model, test_loader, device)
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_loss:.4f} | Train Dice: {avg_dice:.4f} | Val Dice: {val_dice:.4f} | Val IoU: {val_iou:.4f}")

        latest_path = save_dir / "latest_model.pth"
        torch.save(model.state_dict(), latest_path)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            if best_ckpt_path and best_ckpt_path.exists():
                best_ckpt_path.unlink()
             
            new_best_name = f"best_dice_{val_dice:.4f}_iou_{val_iou:.4f}.pth"
            best_ckpt_path = save_dir / new_best_name
            torch.save(model.state_dict(), best_ckpt_path)

if __name__ == "__main__":
    main()
