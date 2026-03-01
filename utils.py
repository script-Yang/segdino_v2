import os
import math
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy.ndimage import distance_transform_edt, binary_erosion

def dice_coeff(pred, target, smooth=1e-5):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    return (2. * intersection + smooth) / (pred.sum() + target.sum() + smooth)

def iou_coeff(pred, target, smooth=1e-5):
    pred = (torch.sigmoid(pred) > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

def iou_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    union = np.logical_or(pred_bin, tgt_bin).sum(dtype=np.float64)
    return float((inter + eps) / (union + eps))

def dice_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray, eps=1e-6) -> float:
    inter = np.logical_and(pred_bin, tgt_bin).sum(dtype=np.float64)
    s = pred_bin.sum(dtype=np.float64) + tgt_bin.sum(dtype=np.float64)
    return float((2.0 * inter + eps) / (s + eps))

def _binary_boundary(mask_bin: np.ndarray) -> np.ndarray:
    return np.logical_and(mask_bin, np.logical_not(binary_erosion(mask_bin)))

def hd95_binary_numpy(pred_bin: np.ndarray, tgt_bin: np.ndarray) -> float:
    pred_has, tgt_has = pred_bin.any(), tgt_bin.any()
    if not pred_has and not tgt_has: return 0.0
    if pred_has != tgt_has: return float("inf")

    Pb, Tb = _binary_boundary(pred_bin), _binary_boundary(tgt_bin)
    if Pb.max() == 0: Pb = pred_bin.astype(bool)
    if Tb.max() == 0: Tb = tgt_bin.astype(bool)

    dist_to_T = distance_transform_edt(np.logical_not(Tb))
    dist_to_P = distance_transform_edt(np.logical_not(Pb))

    Py, Px = np.where(Pb)
    Ty, Tx = np.where(Tb)
    if len(Py) == 0 or len(Ty) == 0: return float("inf")

    d_all = np.concatenate([dist_to_T[Py, Px], dist_to_P[Ty, Tx]], axis=0)
    if d_all.size == 0: return float("inf")
    return float(np.percentile(d_all, 95))

def tensor_to_pil(img_t: torch.Tensor) -> Image.Image:
    img = img_t.detach().cpu().float()
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = img * std + mean
    img = img.clamp(0, 1).numpy()
    img = (img * 255.0).round().astype(np.uint8)
    return Image.fromarray(np.transpose(img, (1, 2, 0)), mode="RGB")

def mask_to_pil(mask_t: torch.Tensor, thr: float = 0.5) -> Image.Image:
    m = mask_t.detach().cpu().float()
    if m.ndim == 3 and m.shape[0] == 1: m = m[0]
    if m.max() > 1.0 or m.min() < 0.0: m = torch.sigmoid(m)
    m_arr = ((m > thr).float() * 255.0).round().clamp(0, 255).byte().numpy()
    return Image.fromarray(m_arr, mode="L")

@torch.no_grad()
def save_eval_visuals(idx, inputs, logits, targets, out_dir, thr=0.5, fname_prefix="test"):
    os.makedirs(out_dir, exist_ok=True)
    img_pil = tensor_to_pil(inputs)
    pred_pil = mask_to_pil(logits, thr)
    gt_pil = mask_to_pil(targets, thr)
    
    base = os.path.join(out_dir, f"{fname_prefix}_{idx:05d}")
    img_pil.save(base + "_img.png")
    pred_pil.save(base + "_pred.png")
    gt_pil.save(base + "_gt.png")

def load_ckpt_flex(model, ckpt_path, map_location="cpu"):
    obj = torch.load(ckpt_path, map_location=map_location, weights_only=True)
    state = obj.get("state_dict", obj) if isinstance(obj, dict) else obj
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing: print("Missing keys:", missing)
    if unexpected: print("Unexpected keys:", unexpected)