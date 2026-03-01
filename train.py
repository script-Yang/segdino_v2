import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from mydataset import FolderDataset, TrainTransform, TestTransform 
from utils import dice_coeff, iou_coeff
from dpt import DPT 

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
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="./kvasir")
    parser.add_argument("--img_size", type=int, default=256)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--dino_size", type=str, default="s", choices=["b", "s"])
    parser.add_argument("--dino_ckpt", type=str, required=True)
    parser.add_argument("--repo_dir", type=str, default="./dinov3")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset_name = os.path.basename(args.data_dir.rstrip("/"))
    args.save_dir = os.path.join(args.save_dir, dataset_name)

    os.makedirs(args.save_dir, exist_ok=True)

    train_dataset = FolderDataset(
        root=args.data_dir,
        split="train",
        transform=TrainTransform(img_size=(args.img_size, args.img_size))
    )
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4)

    test_dataset = FolderDataset(
        root=args.data_dir,
        split="test",
        transform=TestTransform(img_size=(args.img_size, args.img_size)) 
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    if args.dino_size == "b":
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vitb16', source='local', weights=args.dino_ckpt)
    else:
        backbone = torch.hub.load(args.repo_dir, 'dinov3_vits16', source='local', weights=args.dino_ckpt)
        
    model = DPT(nclass=1, backbone=backbone).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.BCEWithLogitsLoss()

    best_val_dice = 0.0
    best_ckpt_path = ""

    for epoch in range(args.epochs):
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

        latest_path = os.path.join(args.save_dir, "latest_model.pth")
        torch.save(model.state_dict(), latest_path)
        
        if val_dice > best_val_dice:
            best_val_dice = val_dice
            if best_ckpt_path and os.path.exists(best_ckpt_path):
                os.remove(best_ckpt_path)
            
            # new_best_name = f"best_ep{epoch+1:02d}_dice_{val_dice:.4f}_iou_{val_iou:.4f}.pth"
            new_best_name = f"best_dice_{val_dice:.4f}_iou_{val_iou:.4f}.pth"
            best_ckpt_path = os.path.join(args.save_dir, new_best_name)
            torch.save(model.state_dict(), best_ckpt_path)

if __name__ == "__main__":
    main()