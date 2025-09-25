import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms

def apply_clahe(img_tensor):
    img = (img_tensor.permute(1,2,0).numpy()*255).astype(np.uint8)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l,a,b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0,tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl,a,b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
    return transforms.ToTensor()(final)

def apply_gamma(img_tensor, gamma=2.2):
    return torch.pow(img_tensor, 1/gamma)

def apply_retinex(img_tensor):
    img = (img_tensor.permute(1,2,0).numpy()*255).astype(np.uint8)
    img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img_retinex = cv2.equalizeHist(img_gray)
    img_retinex = cv2.cvtColor(img_retinex, cv2.COLOR_GRAY2RGB)
    return transforms.ToTensor()(img_retinex)

def apply_zerodcepp(img_tensor, model, device):
    """Apply Zero-DCE++ enhancement using trained model"""
    with torch.no_grad():
        # Ensure input is in correct format [1, 3, H, W]
        if img_tensor.dim() == 3:
            img_tensor = img_tensor.unsqueeze(0)
        
        img_tensor = img_tensor.to(device)
        
        # Forward pass through Zero-DCE++ model
        enhanced_img, _ = model(img_tensor)
        
        # Return as [3, H, W] tensor
        return enhanced_img.squeeze(0).cpu()