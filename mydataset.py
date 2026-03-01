import os
import random
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
from torchvision.transforms import InterpolationMode, ColorJitter

# class TrainTransform:
#     def __init__(self, img_size=(256, 256)):
#         self.img_size = img_size
        
#     def __call__(self, img, mask):
#         img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)
#         mask = TF.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)
#         img = TF.to_tensor(img) 
#         mask = TF.to_tensor(mask)
#         img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
#         return img, mask

class TrainTransform:
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size
        self.color_jitter = ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)
        
    def __call__(self, img, mask):
        img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)
        
        if random.random() > 0.5:
            img = TF.hflip(img)
            mask = TF.hflip(mask)
            
        if random.random() > 0.5:
            img = TF.vflip(img)
            mask = TF.vflip(mask)
            
        if random.random() > 0.5:
            angle = random.randint(-15, 15)
            img = TF.rotate(img, angle, interpolation=InterpolationMode.BILINEAR)
            mask = TF.rotate(mask, angle, interpolation=InterpolationMode.NEAREST)

        if random.random() > 0.5:
            img = self.color_jitter(img)

        img = TF.to_tensor(img) 
        mask = TF.to_tensor(mask)
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img, mask

class TestTransform:
    def __init__(self, img_size=(256, 256)):
        self.img_size = img_size

    def __call__(self, img, mask):
        img = TF.resize(img, self.img_size, interpolation=InterpolationMode.BILINEAR)
        mask = TF.resize(mask, self.img_size, interpolation=InterpolationMode.NEAREST)

        img = TF.to_tensor(img)
        mask = TF.to_tensor(mask)
        
        img = TF.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        return img, mask

class FolderDataset(Dataset):
    def __init__(self, root, split="train", img_dir_name="image", label_dir_name="mask", transform=None):
        self.img_dir = os.path.join(root, split, img_dir_name)
        self.mask_dir = os.path.join(root, split, label_dir_name)
        
        self.img_names = sorted([f for f in os.listdir(self.img_dir) if not f.startswith('.')])
        self.mask_names = sorted([f for f in os.listdir(self.mask_dir) if not f.startswith('.')])
        
        assert len(self.img_names) == len(self.mask_names)
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_names[idx])
        mask_path = os.path.join(self.mask_dir, self.mask_names[idx])
        
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")
        
        if self.transform is not None:
            img, mask = self.transform(img, mask)
            
        return img, mask, self.img_names[idx]