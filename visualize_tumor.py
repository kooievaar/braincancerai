#!/usr/bin/env python3
"""
Visualization script for Brain Tumor Segmentation
Generates PNG images showing MRI scans with tumor regions highlighted.
Uses ground truth BraTS segmentations for accurate tumor boundaries.

Usage:
    python visualize_tumor.py              # Process all datasets
    python visualize_tumor.py --train      # Training set only
    python visualize_tumor.py --test       # Test set only
"""

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import os
import glob

# Configuration
DETECTION_THRESHOLD = 244.7

# Dataset paths (relative to project root or absolute)
TRAIN_PATH = 'mri-scans/UCSD-PTGBM training MRI-data 1'
TEST_PATH = 'mri-scans/UCSD-PTGBM-BraTS-2024-test-set MRI-data 2'

OUTPUT_TRAIN = 'tumor_output'
OUTPUT_TEST = 'tumor_output_test'


def create_segmentation_visualization(mri_path, seg_path, output_path, patient_name):
    """
    Create proper visualization using ground truth tumor segmentation.
    Color coding:
    - Yellow = Edema (swelling)
    - Red = Enhancing Tumor (active cancer)
    - Dark = Necrosis (dead tissue)
    """
    # Load MRI and ground truth segmentation
    mri = nib.load(mri_path).get_fdata()
    seg = nib.load(seg_path).get_fdata()
    
    # BraTS labels: 0=background, 1=edema, 2=non-enhancing, 3=enhancing, 4=necrosis
    tumor_mask = (seg > 0).astype(float)
    edema = (seg == 1).astype(float)
    enhancing = (seg == 3).astype(float)
    necrosis = (seg == 4).astype(float)
    
    # Create figure
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
        
        # Row 2: Ground Truth Tumor with color coding
        axes[1, idx].imshow(mri_norm, cmap='gray', origin='lower')
        
        if np.any(tumor_slice):
            overlay = np.zeros((*tumor_slice.shape, 4))
            
            # Edema (yellow)
            overlay[edema_slice > 0, 0] = 1.0
            overlay[edema_slice > 0, 1] = 1.0
            overlay[edema_slice > 0, 2] = 0.0
            overlay[edema_slice > 0, 3] = 0.5
            
            # Enhancing tumor (red)
            overlay[enhancing_slice > 0, 0] = 1.0
            overlay[enhancing_slice > 0, 1] = 0.0
            overlay[enhancing_slice > 0, 2] = 0.0
            overlay[enhancing_slice > 0, 3] = 0.8
            
            # Necrosis (dark)
            overlay[nec_slice > 0, 0] = 0.5
            overlay[nec_slice > 0, 1] = 0.0
            overlay[nec_slice > 0, 2] = 0.0
            overlay[nec_slice > 0, 3] = 0.9
            
            axes[1, idx].imshow(overlay, origin='lower')
        
        tumor_vox = int(np.sum(tumor_slice))
        axes[1, idx].set_title(f'{view_name} - Tumor: {tumor_vox:,} voxels', fontsize=10)
        axes[1, idx].axis('off')
        
        # Row 3: Binary mask
        axes[2, idx].imshow(tumor_slice, cmap='Reds', origin='lower')
        axes[2, idx].set_title(f'{view_name} - Segmentation', fontsize=10)
        axes[2, idx].axis('off')
    
    # Statistics
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
        'total': int(total_tumor),
        'edema': int(edema_vol),
        'enhancing': int(enhancing_vol),
        'necrosis': int(nec_vol)
    }


def process_dataset(data_path, output_dir, dataset_name):
    """Process all patients in a dataset."""
    os.makedirs(output_dir, exist_ok=True)
    
    patients = sorted([d for d in os.listdir(data_path) if d.startswith('UCSD-PTGBM')])
    print(f"Processing {len(patients)} patients from {dataset_name}...")
    
    results = []
    for i, patient in enumerate(patients):
        mri_file = os.path.join(data_path, patient, patient + '_T1pre.nii.gz')
        seg_file = os.path.join(data_path, patient, patient + '_BraTS_tumor_seg.nii.gz')
        out_file = os.path.join(output_dir, patient + '_segmentation.png')
        
        if os.path.exists(mri_file) and os.path.exists(seg_file):
            try:
                stats = create_segmentation_visualization(mri_file, seg_file, out_file, patient)
                results.append(stats)
                print(f"  {patient}: {stats['total']:,} tumor voxels")
            except Exception as e:
                print(f"  Error {patient}: {e}")
        
        if (i+1) % 20 == 0:
            print(f"  Progress: {i+1}/{len(patients)}")
    
    return results


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate tumor segmentation visualizations')
    parser.add_argument('--train', action='store_true', help='Process training set')
    parser.add_argument('--test', action='store_true', help='Process test set')
    parser.add_argument('--all', action='store_true', help='Process all datasets (default)')
    
    args = parser.parse_args()
    
    # Default: process all
    do_train = args.train or args.all
    do_test = args.test or args.all
    
    print("=" * 60)
    print("BRAIN TUMOR SEGMENTATION VISUALIZATION")
    print("=" * 60)
    
    if do_train:
        print(f"\n[Training Set]")
        results = process_dataset(TRAIN_PATH, OUTPUT_TRAIN, "Training")
        print(f"Completed: {len(results)} visualizations")
    
    if do_test:
        print(f"\n[Test Set]")
        results = process_dataset(TEST_PATH, OUTPUT_TEST, "Test")
        print(f"Completed: {len(results)} visualizations")
    
    print("\n" + "=" * 60)
    print("VISUALIZATION COMPLETE")
    print("=" * 60)
    print(f"\nOutput directories:")
    if do_train:
        print(f"  Training: {OUTPUT_TRAIN}/")
    if do_test:
        print(f"  Test: {OUTPUT_TEST}/")


if __name__ == '__main__':
    main()