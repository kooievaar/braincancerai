#!/usr/bin/env python3
"""
Brain Tumor Segmentation using Deep Learning (U-Net)
Proper segmentation with Dice score optimization
"""

import numpy as np
import nibabel as nib
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from scipy.ndimage import zoom

# Configuration
TRAIN_PATH = 'mri-scans/UCSD-PTGBM training MRI-data 1'
IMG_SIZE = 128  # Smaller for memory efficiency
BATCH_SIZE = 4
EPOCHS = 30
LEARNING_RATE = 0.001

print("="*60)
print("BRAIN TUMOR SEGMENTATION - U-NET DEEP LEARNING")
print("="*60)

# Dataset class
class BrainMRIDataset(Dataset):
    def __init__(self, mri_paths, seg_paths, augment=False):
        self.mri_paths = mri_paths
        self.seg_paths = seg_paths
        self.augment = augment
    
    def __len__(self):
        return len(self.mri_paths)
    
    def __getitem__(self, idx):
        # Load MRI
        mri = nib.load(self.mri_paths[idx]).get_fdata()
        seg = nib.load(self.seg_paths[idx]).get_fdata()
        
        # Normalize MRI to 0-1
        mri = (mri - mri.min()) / (mri.max() - mri.min() + 1e-8)
        
        # Resize to standard size
        factors = (IMG_SIZE/mri.shape[0], IMG_SIZE/mri.shape[1], IMG_SIZE/mri.shape[2])
        mri = zoom(mri, factors, order=1)
        seg = zoom(seg, factors, order=0)
        
        # Binarize segmentation (0 or 1)
        seg = (seg > 0).astype(np.float32)
        
        # Add channel dimension: (1, D, H, W)
        mri = np.expand_dims(mri, axis=0)
        seg = np.expand_dims(seg, axis=0)
        
        return torch.FloatTensor(mri), torch.FloatTensor(seg)


# U-Net Architecture
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super().__init__()
        
        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.pool1 = nn.MaxPool3d(2)
        
        self.enc2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool3d(2)
        
        self.enc3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool3d(2)
        
        self.enc4 = DoubleConv(128, 256)
        
        # Decoder
        self.up4 = nn.ConvTranspose3d(256, 128, 2, stride=2)
        self.dec4 = DoubleConv(256, 128)
        
        self.up3 = nn.ConvTranspose3d(128, 64, 2, stride=2)
        self.dec3 = DoubleConv(128, 64)
        
        self.up2 = nn.ConvTranspose3d(64, 32, 2, stride=2)
        self.dec2 = DoubleConv(64, 32)
        
        # Output
        self.out = nn.Conv3d(32, out_channels, 1)
    
    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))
        
        # Decoder
        d4 = self.up4(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))
        
        d3 = self.up3(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))
        
        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))
        
        return torch.sigmoid(self.out(d2))


# Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, pred, target):
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        
        intersection = (pred_flat * target_flat).sum()
        dice = (2. * intersection + self.smooth) / (pred_flat.sum() + target_flat.sum() + self.smooth)
        
        return 1 - dice


# Metrics
def calculate_dice(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    dice = (2. * intersection + 1e-6) / (pred_binary.sum() + target.sum() + 1e-6)
    return dice.item()


def calculate_iou(pred, target, threshold=0.5):
    pred_binary = (pred > threshold).float()
    intersection = (pred_binary * target).sum()
    union = pred_binary.sum() + target.sum() - intersection
    return (intersection / (union + 1e-6)).item()


# Load data
print("\n[1] Loading dataset...")
mri_files = []
seg_files = []

patients = sorted([d for d in os.listdir(TRAIN_PATH) if d.startswith('UCSD-PTGBM')])
print(f"Found {len(patients)} patients")

for patient in patients:
    t1_path = os.path.join(TRAIN_PATH, patient, patient + '_T1pre.nii.gz')
    seg_path = os.path.join(TRAIN_PATH, patient, patient + '_BraTS_tumor_seg.nii.gz')
    
    if os.path.exists(t1_path) and os.path.exists(seg_path):
        mri_files.append(t1_path)
        seg_files.append(seg_path)

print(f"Valid samples: {len(mri_files)}")

# Split data
train_mri, val_mri, train_seg, val_seg = train_test_split(
    mri_files, seg_files, test_size=0.2, random_state=42
)

print(f"Training: {len(train_mri)}, Validation: {len(val_mri)}")

# Create datasets
train_dataset = BrainMRIDataset(train_mri, train_seg)
val_dataset = BrainMRIDataset(val_mri, val_seg)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Model
print("\n[2] Creating U-Net model...")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = UNet3D().to(device)
criterion = DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)

# Training
print("\n[3] Training U-Net...")
best_dice = 0

for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    train_dice = 0
    
    for batch_mri, batch_seg in train_loader:
        batch_mri = batch_mri.to(device)
        batch_seg = batch_seg.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_mri)
        loss = criterion(outputs, batch_seg)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_dice += calculate_dice(outputs, batch_seg)
    
    # Validation
    model.eval()
    val_dice = 0
    val_iou = 0
    
    with torch.no_grad():
        for batch_mri, batch_seg in val_loader:
            batch_mri = batch_mri.to(device)
            batch_seg = batch_seg.to(device)
            
            outputs = model(batch_mri)
            val_dice += calculate_dice(outputs, batch_seg)
            val_iou += calculate_iou(outputs, batch_seg)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    avg_val_dice = val_dice / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    
    scheduler.step(avg_train_loss)
    
    if avg_val_dice > best_dice:
        best_dice = avg_val_dice
        torch.save(model.state_dict(), 'pretrained_unet_model.pth')
    
    if (epoch + 1) % 5 == 0:
        print(f"Epoch {epoch+1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f}, Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}, Val IoU: {avg_val_iou:.4f}")

print(f"\nBest Validation Dice: {best_dice:.4f}")

# Load best model
model.load_state_dict(torch.load('pretrained_unet_model.pth'))

# Test on sample
print("\n[4] Testing on sample patient...")
model.eval()

sample_mri = nib.load(val_mri[0]).get_fdata()
sample_seg = nib.load(val_seg[0]).get_fdata()

# Resize
factors = (IMG_SIZE/sample_mri.shape[0], IMG_SIZE/sample_mri.shape[1], IMG_SIZE/sample_mri.shape[2])
sample_mri_resized = zoom(sample_mri, factors, order=1)
sample_mri_norm = (sample_mri_resized - sample_mri_resized.min()) / (sample_mri_resized.max() - sample_mri_resized.min() + 1e-8)
sample_input = torch.FloatTensor(np.expand_dims(sample_mri_norm, (0, 1))).to(device)

with torch.no_grad():
    pred = model(sample_input).cpu().numpy()[0, 0]

# Calculate metrics
pred_binary = (pred > 0.5).astype(float)
actual = zoom(sample_seg, factors, order=0)
actual_binary = (actual > 0).astype(float)

dice = calculate_dice(torch.FloatTensor(pred_binary), torch.FloatTensor(actual_binary))
iou = calculate_iou(torch.FloatTensor(pred_binary), torch.FloatTensor(actual_binary))

print(f"Sample Prediction - Dice: {dice:.4f}, IoU: {iou:.4f}")

# Compare with old method
old_threshold = 244.7
old_pred = sample_mri_resized > old_threshold
old_dice = calculate_dice(torch.FloatTensor(old_pred.astype(float)), torch.FloatTensor(actual_binary))

print(f"Old method (threshold) - Dice: {old_dice:.4f}")
print(f"Improvement: {(dice - old_dice)*100:.1f}%")

print("\n" + "="*60)
print("TRAINING COMPLETE")
print("="*60)
print(f"Final U-Net Dice Score: {best_dice:.4f}")
print("Model saved to: pretrained_unet_model.pth")