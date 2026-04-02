#!/usr/bin/env python3
"""Analyze sample_mri.nii MRI data"""
import sys
import nibabel as nib
import numpy as np

# Get filename from command line or default
filename = sys.argv[1] if len(sys.argv) > 1 else 'sample_mri.nii'

# Load the NIfTI file
img = nib.load(filename)
data = img.get_fdata()
header = img.header

print("=" * 60)
print("MRI DATA ANALYSIS")
print("=" * 60)

# Basic info
print(f"\nFile: {filename}")
print(f"Dimensions: {data.shape}")
print(f"Data type: {data.dtype}")
print(f"Memory size: {data.nbytes / 1024:.2f} KB")

# Header information
print(f"\nNIfTI Header:")
print(f"  Description: {header.get('descrip', b'').decode('utf-8', errors='ignore') if header.get('descrip') else 'None'}")
print(f"  Intent code: {header.get('intent_code', 'N/A')}")
print(f"  Data scale slope: {header.get('scl_slope', 1.0)}")
print(f"  Data scale inter: {header.get('scl_inter', 0.0)}")

# Value statistics
print(f"\nValue Statistics:")
print(f"  Min value: {data.min():.4f}")
print(f"  Max value: {data.max():.4f}")
print(f"  Mean: {data.mean():.4f}")
print(f"  Std: {data.std():.4f}")

# Count non-zero voxels
nonzero = np.count_nonzero(data)
total = data.size
print(f"  Non-zero voxels: {nonzero:,} / {total:,} ({100*nonzero/total:.1f}%)")

# Find threshold for brain mask (simple approach)
threshold = 0.1
brain_mask = data > threshold
brain_voxels = np.count_nonzero(brain_mask)
print(f"\nBrain Mask (threshold > {threshold}):")
print(f"  Brain voxels: {brain_voxels:,}")
pixdim = header.get('pixdim', [1, 1, 1, 1])
pixel_size = pixdim[1] ** 3 if len(pixdim) > 1 else 1
brain_volume = brain_voxels * pixel_size
print(f"  Brain volume estimate: {brain_volume:.2f} mm³")

# Histogram bins
hist, bins = np.histogram(data[data > 0].flatten(), bins=20)
print(f"\nHistogram of non-zero values:")
for i in range(len(hist)):
    print(f"  [{bins[i]:.2f} - {bins[i+1]:.2f}]: {hist[i]:,} voxels")

print("\n" + "=" * 60)
print("Analysis complete!")
print("=" * 60)
