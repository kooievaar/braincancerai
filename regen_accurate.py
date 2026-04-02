#!/usr/bin/env python3
"""
Proper Brain Tumor Segmentation Visualization
Uses actual BraTS ground truth - NOT intensity threshold!
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

# Configuration - use relative paths (works after git clone)
TEST_PATH = 'mri-scans/UCSD-PTGBM-BraTS-2024-test-set MRI-data 2'
TRAIN_PATH = 'mri-scans/UCSD-PTGBM training MRI-data 1'
OUTPUT_TEST = 'tumor_output_test'
OUTPUT_TRAIN = 'tumor_output'

os.makedirs(OUTPUT_TEST, exist_ok=True)
os.makedirs(OUTPUT_TRAIN, exist_ok=True)


def create_accurate_visualization(mri_path, seg_path, output_path, patient_name):
    """
    Create visualization using ACTUAL ground truth segmentation.
    The red color shows ONLY the actual tumor, not the whole head!
    """
    # Load MRI and ground truth segmentation from BraTS
    mri = nib.load(mri_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    
    # BraTS labels: 0=background, 1=edema, 2=non-enhancing, 3=enhancing, 4=necrosis
    # Use actual ground truth - this is what radiologists marked!
    tumor_actual = (seg > 0).astype(float)  # All tumor types combined
    
    # Get specific tumor types for color coding
    edema = (seg == 1).astype(float)      # Swelling - yellow
    non_enh = (seg == 2).astype(float)    # Non-enhancing tumor - orange
    enhancing = (seg == 3).astype(float) # Enhancing tumor - RED
    necrosis = (seg == 4).astype(float)  # Dead tissue - dark red
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 10))
    fig.patch.set_facecolor('white')
    
    # Three orthogonal slices
    slices = [
        (0, mri.shape[0] // 2, 'Axial'),
        (1, mri.shape[1] // 2, 'Coronal'),
        (2, mri.shape[2] // 2, 'Sagittal')
    ]
    
    for idx, (axis, slice_idx, view_name) in enumerate(slices):
        if axis == 0:
            mri_slice = mri[slice_idx, :, :]
            tumor_slice = tumor_actual[slice_idx, :, :]
            edema_s = edema[slice_idx, :, :]
            non_enh_s = non_enh[slice_idx, :, :]
            enh_s = enhancing[slice_idx, :, :]
            nec_s = necrosis[slice_idx, :, :]
        elif axis == 1:
            mri_slice = mri[:, slice_idx, :]
            tumor_slice = tumor_actual[:, slice_idx, :]
            edema_s = edema[:, slice_idx, :]
            non_enh_s = non_enh[:, slice_idx, :]
            enh_s = enhancing[:, slice_idx, :]
            nec_s = necrosis[:, slice_idx, :]
        else:
            mri_slice = mri[:, :, slice_idx]
            tumor_slice = tumor_actual[:, :, slice_idx]
            edema_s = edema[:, :, slice_idx]
            non_enh_s = non_enh[:, :, slice_idx]
            enh_s = enhancing[:, :, slice_idx]
            nec_s = necrosis[:, :, slice_idx]
        
        # Normalize MRI for display
        mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
        
        # Row 1: Original MRI (grayscale)
        axes[0, idx].imshow(mri_norm, cmap='gray', origin='lower')
        axes[0, idx].set_title(f'{view_name} - Original MRI', fontsize=10)
        axes[0, idx].axis('off')
        
        # Row 2: MRI with ACTUAL tumor overlay (from ground truth!)
        axes[1, idx].imshow(mri_norm, cmap='gray', origin='lower')
        
        if np.any(tumor_slice):
            # Create RGBA overlay with actual tumor regions
            overlay = np.zeros((*tumor_slice.shape, 4))
            
            # Color each tumor type differently
            # Edema (yellow) - swelling/edema
            overlay[edema_s > 0, 0] = 1.0
            overlay[edema_s > 0, 1] = 1.0  
            overlay[edema_s > 0, 2] = 0.0
            overlay[edema_s > 0, 3] = 0.4
            
            # Non-enhancing tumor (orange)
            overlay[non_enh_s > 0, 0] = 1.0
            overlay[non_enh_s > 0, 1] = 0.5
            overlay[non_enh_s > 0, 2] = 0.0
            overlay[non_enh_s > 0, 3] = 0.7
            
            # Enhancing tumor (RED) - the dangerous active cancer!
            overlay[enh_s > 0, 0] = 1.0
            overlay[enh_s > 0, 1] = 0.0
            overlay[enh_s > 0, 2] = 0.0
            overlay[enh_s > 0, 3] = 0.9
            
            # Necrosis (dark) - dead tissue
            overlay[nec_s > 0, 0] = 0.3
            overlay[nec_s > 0, 1] = 0.0
            overlay[nec_s > 0, 2] = 0.0
            overlay[nec_s > 0, 3] = 0.9
            
            axes[1, idx].imshow(overlay, origin='lower')
        
        tumor_vox = int(np.sum(tumor_slice))
        axes[1, idx].set_title(f'{view_name} - Tumor: {tumor_vox:,} voxels (REAL)', fontsize=10)
        axes[1, idx].axis('off')
    
    # Calculate actual tumor stats from ground truth
    total_tumor = np.sum(tumor_actual)
    e = np.sum(edema)
    ne = np.sum(non_enh)
    en = np.sum(enhancing)
    n = np.sum(necrosis)
    
    plt.suptitle(f'{patient_name}\nEdema:{e:,} Non-Enh:{ne:,} Enh(RED):{en:,} Necrosis:{n:,} Total:{total_tumor:,}', 
                 fontsize=11, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=100, bbox_inches='tight', facecolor='white')
    plt.close()
    
    return {'total': int(total_tumor), 'enhancing': int(en)}


def process_dataset(data_path, output_dir, name):
    """Process all patients in a dataset."""
    patients = sorted([d for d in os.listdir(data_path) if d.startswith('UCSD-PTGBM')])
    print(f"Processing {len(patients)} patients from {name}...")
    
    results = []
    for i, patient in enumerate(patients):
        mri_file = os.path.join(data_path, patient, patient + '_T1pre.nii.gz')
        seg_file = os.path.join(data_path, patient, patient + '_BraTS_tumor_seg.nii.gz')
        out_file = os.path.join(output_dir, patient + '_visualization.png')
        
        if os.path.exists(mri_file) and os.path.exists(seg_file):
            try:
                stats = create_accurate_visualization(mri_file, seg_file, out_file, patient)
                results.append(stats)
                if (i+1) % 10 == 0:
                    print(f"  {i+1}/{len(patients)}")
            except Exception as e:
                print(f"  Error {patient}: {e}")
    
    print(f"  Done: {len(results)} images")
    return results


if __name__ == '__main__':
    print("="*60)
    print("GENERATING ACCURATE TUMOR VISUALIZATIONS")
    print("Using ACTUAL ground truth from BraTS - not threshold!")
    print("="*60)
    
    # Process test set
    results_test = process_dataset(TEST_PATH, OUTPUT_TEST, "Test")
    
    # Process training set  
    results_train = process_dataset(TRAIN_PATH, OUTPUT_TRAIN, "Training")
    
    print("\n" + "="*60)
    print("DONE!")
    print("="*60)
    print(f"Test: {len(results_test)} images")
    print(f"Training: {len(results_train)} images")