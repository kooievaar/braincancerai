#!/usr/bin/env python3
"""
Proper Brain Tumor Segmentation Visualization
Uses ground truth BraTS segmentations for accurate tumor boundaries
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os

# Paths
TRAIN_PATH = r'H:\__MRI\braincancerai\mri-scans\UCSD-PTGBM training MRI-data 1'
TEST_PATH = r'H:\__MRI\braincancerai\mri-scans\UCSD-PTGBM-BraTS-2024-test-set MRI-data 2'
TRAIN_OUT = 'tumor_output'
TEST_OUT = 'tumor_output_test'

os.makedirs(TRAIN_OUT, exist_ok=True)
os.makedirs(TEST_OUT, exist_ok=True)

def create_proper_segmentation(mri_path, seg_path, output_path, patient_name):
    """Create proper visualization using actual tumor segmentation"""
    
    # Load MRI and ground truth segmentation
    mri = nib.load(mri_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    
    # BraTS labels: 0=background, 1=edema, 2=non-enhancing, 3=enhancing, 4=necrosis
    # We'll show all tumor regions (labels 1-4) in RED
    tumor_mask = (seg > 0).astype(float)
    
    # Get specific tumor types for visualization
    edema = (seg == 1).astype(float)
    enhancing = (seg == 3).astype(float)
    necrosis = (seg == 4).astype(float)
    
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))
    fig.patch.set_facecolor('white')
    
    # Three orthogonal views
    slices = [
        (0, mri.shape[0] // 2, 'Axial'),
        (1, mri.shape[1] // 2, 'Coronal'),
        (2, mri.shape[2] // 2, 'Sagittal')
    ]
    
    for idx, (axis, slice_idx, view_name) in enumerate(slices):
        # Get 2D slices
        if axis == 0:
            mri_slice = mri[slice_idx, :, :]
            tumor_slice = tumor_mask[slice_idx, :, :]
            edema_slice = edema[slice_idx, :, :]
            enhancing_slice = enhancing[slice_idx, :, :]
            nec_slice = necrosis[slice_idx, :, :]
        elif axis == 1:
            mri_slice = mri[:, slice_idx, :]
            tumor_slice = tumor_mask[:, slice_idx, :]
            edema_slice = edema[:, slice_idx, :]
            enhancing_slice = enhancing[:, slice_idx, :]
            nec_slice = necrosis[:, slice_idx, :]
        else:
            mri_slice = mri[:, :, slice_idx]
            tumor_slice = tumor_mask[:, :, slice_idx]
            edema_slice = edema[:, :, slice_idx]
            enhancing_slice = enhancing[:, :, slice_idx]
            nec_slice = necrosis[:, :, slice_idx]
        
        # Normalize MRI
        mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
        
        # Row 1: Original MRI
        axes[0, idx].imshow(mri_norm, cmap='gray', origin='lower')
        axes[0, idx].set_title(f'{view_name} - T1 MRI', fontsize=10)
        axes[0, idx].axis('off')
        
        # Row 2: Ground Truth Tumor
        axes[1, idx].imshow(mri_norm, cmap='gray', origin='lower')
        
        # Create multi-color overlay for different tumor types
        if np.any(tumor_slice):
            overlay = np.zeros((*tumor_slice.shape, 4))
            
            # Edema (yellow)
            overlay[edema_slice > 0, 0] = 1.0
            overlay[edema_slice > 0, 1] = 1.0
            overlay[edema_slice > 0, 2] = 0.0
            overlay[edema_slice > 0, 3] = 0.5
            
            # Enhancing tumor (RED)
            overlay[enhancing_slice > 0, 0] = 1.0
            overlay[enhancing_slice > 0, 1] = 0.0
            overlay[enhancing_slice > 0, 2] = 0.0
            overlay[enhancing_slice > 0, 3] = 0.8
            
            # Necrosis (dark red/black)
            overlay[nec_slice > 0, 0] = 0.5
            overlay[nec_slice > 0, 1] = 0.0
            overlay[nec_slice > 0, 2] = 0.0
            overlay[nec_slice > 0, 3] = 0.9
            
            axes[1, idx].imshow(overlay, origin='lower')
        
        tumor_vox = int(np.sum(tumor_slice))
        axes[1, idx].set_title(f'{view_name} - Tumor (RED={tumor_vox:,} voxels)', fontsize=10)
        axes[1, idx].axis('off')
        
        # Row 3: Binary mask only
        axes[2, idx].imshow(tumor_slice, cmap='Reds', origin='lower')
        axes[2, idx].set_title(f'{view_name} - Segmentation', fontsize=10)
        axes[2, idx].axis('off')
    
    # Calculate stats
    total_tumor = np.sum(tumor_mask)
    edema_vol = np.sum(edema)
    enhancing_vol = np.sum(enhancing)
    nec_vol = np.sum(necrosis)
    
    plt.suptitle(f'Patient: {patient_name}\nEdema(Yellow): {edema_vol:,} | Enhancing(Red): {enhancing_vol:,} | Necrosis(Dark): {nec_vol:,} | Total: {total_tumor:,} voxels', 
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {
        'total': total_tumor,
        'edema': edema_vol,
        'enhancing': enhancing_vol,
        'necrosis': nec_vol
    }


print("="*60)
print("PROPER TUMOR SEGMENTATION VISUALIZATION")
print("Using Ground Truth BraTS Segmentations")
print("="*60)

# Process training set
print("\n[1] Processing training set...")
train_patients = sorted([d for d in os.listdir(TRAIN_PATH) if d.startswith('UCSD-PTGBM')])

train_results = []
for i, patient in enumerate(train_patients):
    mri_file = os.path.join(TRAIN_PATH, patient, patient + '_T1pre.nii.gz')
    seg_file = os.path.join(TRAIN_PATH, patient, patient + '_BraTS_tumor_seg.nii.gz')
    out_file = os.path.join(TRAIN_OUT, patient + '_segmentation.png')
    
    if os.path.exists(mri_file) and os.path.exists(seg_file):
        stats = create_proper_segmentation(mri_file, seg_file, out_file, patient)
        train_results.append(stats)
        
    if (i+1) % 20 == 0:
        print(f"  Processed {i+1}/{len(train_patients)}")

print(f"Training: {len(train_results)} visualizations")

# Process test set  
print("\n[2] Processing test set...")
test_patients = sorted([d for d in os.listdir(TEST_PATH) if d.startswith('UCSD-PTGBM')])

test_results = []
for i, patient in enumerate(test_patients):
    mri_file = os.path.join(TEST_PATH, patient, patient + '_T1pre.nii.gz')
    seg_file = os.path.join(TEST_PATH, patient, patient + '_BraTS_tumor_seg.nii.gz')
    out_file = os.path.join(TEST_OUT, patient + '_segmentation.png')
    
    if os.path.exists(mri_file) and os.path.exists(seg_file):
        stats = create_proper_segmentation(mri_file, seg_file, out_file, patient)
        test_results.append(stats)

print(f"Test: {len(test_results)} visualizations")

# Calculate statistics
print("\n" + "="*60)
print("SEGMENTATION STATISTICS")
print("="*60)

train_total = sum(r['total'] for r in train_results)
test_total = sum(r['total'] for r in test_results)

print(f"\nTraining Set ({len(train_results)} patients):")
print(f"  Average tumor voxels: {train_total/len(train_results):,.0f}")
print(f"  Total tumor voxels: {train_total:,}")

print(f"\nTest Set ({len(test_results)} patients):")
print(f"  Average tumor voxels: {test_total/len(test_results):,.0f}")
print(f"  Total tumor voxels: {test_total:,}")

print("\n" + "="*60)
print("COLOR CODING:")
print("  Yellow = Edema (swelling)")
print("  Bright Red = Enhancing Tumor (active cancer)")
print("  Dark Red/Black = Necrosis (dead tissue)")
print("="*60)