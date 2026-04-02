#!/usr/bin/env python3
"""
Brain Tumor Detection and Segmentation using U-Net Deep Learning
This is the main detection system for the braincancerai project.
Uses ground truth BraTS segmentations for training and evaluation.
"""

import nibabel as nib
import numpy as np
import os
import sys

# Configuration
MODEL_PATH = 'pretrained_unet_model.pth'  # When trained
DETECTION_THRESHOLD = 244.7  # Fallback to threshold method if no model

def load_mri(mri_path):
    """Load and preprocess MRI scan."""
    img = nib.load(mri_path)
    data = img.get_fdata()
    # Normalize to 0-1
    data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
    return data_norm, img.affine, img.header


def detect_tumor_deep_learning(mri_path, model_path=MODEL_PATH):
    """
    Detect tumors using U-Net deep learning model.
    Requires trained model file.
    """
    try:
        import torch
        from train_unet import LightUNet
        
        # Load model
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = LightUNet().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        
        # Load and preprocess MRI
        data, _, _ = load_mri(mri_path)
        
        # Resize to model input size (64x64x64)
        from scipy.ndimage import zoom
        factors = (64/data.shape[0], 64/data.shape[1], 64/data.shape[2])
        data_resized = zoom(data, factors, order=1)
        data_input = torch.FloatTensor(np.expand_dims(data_resized, (0, 1))).to(device)
        
        # Predict
        with torch.no_grad():
            pred = model(data_input).cpu().numpy()[0, 0]
        
        # Binarize
        tumor_mask = pred > 0.5
        return tumor_mask, pred
        
    except Exception as e:
        print(f"Deep learning model not available: {e}")
        return None, None


def detect_tumor_threshold(mri_path, threshold=DETECTION_THRESHOLD):
    """
    Detect tumors using intensity threshold (fallback method).
    This was the initial approach - less accurate than U-Net.
    """
    data, _, _ = load_mri(mri_path)
    
    # Normalize threshold based on data range
    data_max = data.max()
    norm_threshold = threshold / data_max if data_max > 0 else 0.15
    
    # Detect high intensity regions (potential tumor)
    tumor_mask = data > norm_threshold
    
    # Filter out very small regions
    from scipy.ndimage import label
    labeled, num_features = label(tumor_mask)
    
    # Keep only regions with more than 100 voxels
    filtered_mask = np.zeros_like(tumor_mask)
    for i in range(1, num_features + 1):
        region_size = np.sum(labeled == i)
        if region_size > 100:
            filtered_mask[labeled == i] = True
    
    return filtered_mask


def analyze_tumor(mri_path, use_deep_learning=True):
    """
    Analyze MRI scan for tumor presence and characteristics.
    
    Args:
        mri_path: Path to T1-weighted MRI NIfTI file
        use_deep_learning: Whether to use U-Net (True) or threshold (False)
    
    Returns:
        dict: Analysis results
    """
    print(f"\nAnalyzing: {os.path.basename(mri_path)}")
    
    if use_deep_learning:
        tumor_mask, prediction = detect_tumor_deep_learning(mri_path)
        if tumor_mask is not None:
            method = "U-Net Deep Learning"
        else:
            tumor_mask = detect_tumor_threshold(mri_path)
            method = "Intensity Threshold (fallback)"
    else:
        tumor_mask = detect_tumor_threshold(mri_path)
        method = "Intensity Threshold"
    
    # Calculate statistics
    tumor_voxels = np.sum(tumor_mask)
    total_voxels = tumor_mask.size
    tumor_ratio = tumor_voxels / total_voxels
    
    # Determine if tumor is present
    has_tumor = tumor_voxels > 100
    
    # Load original MRI for additional stats
    data, _, _ = load_mri(mri_path)
    
    results = {
        'file': mri_path,
        'method': method,
        'has_tumor': has_tumor,
        'tumor_voxels': int(tumor_voxels),
        'total_voxels': total_voxels,
        'tumor_ratio': float(tumor_ratio),
        'mean_intensity': float(data.mean()),
        'max_intensity': float(data.max()),
    }
    
    # Print results
    print(f"  Method: {method}")
    print(f"  Tumor Detected: {'YES' if has_tumor else 'NO'}")
    print(f"  Tumor Voxels: {tumor_voxels:,}")
    print(f"  Tumor Ratio: {tumor_ratio*100:.2f}%")
    
    return results


def visualize_result(mri_path, tumor_mask, output_path=None):
    """Generate visualization of detection result."""
    import matplotlib.pyplot as plt
    
    data, _, _ = load_mri(mri_path)
    
    # Create figure with 3 views
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    slices = [(0, data.shape[0]//2), (1, data.shape[1]//2), (2, data.shape[2]//2)]
    
    for idx, (ax, sl) in enumerate(slices):
        if ax == 0:
            mri_slice = data[sl, :, :]
            mask_slice = tumor_mask[sl, :, :] if tumor_mask is not None else np.zeros_like(data[sl,:,:])
        elif ax == 1:
            mri_slice = data[:, sl, :]
            mask_slice = tumor_mask[:, sl, :] if tumor_mask is not None else np.zeros_like(data[:,sl,:])
        else:
            mri_slice = data[:, :, sl]
            mask_slice = tumor_mask[:, :, sl] if tumor_mask is not None else np.zeros_like(data[:,:,sl])
        
        mri_norm = (mri_slice - mri_slice.min()) / (mri_slice.max() - mri_slice.min() + 1e-8)
        
        axes[0, idx].imshow(mri_norm, cmap='gray', origin='lower')
        axes[0, idx].set_title(f'View {idx+1}')
        axes[0, idx].axis('off')
        
        axes[1, idx].imshow(mri_norm, cmap='gray', origin='lower')
        if np.any(mask_slice):
            overlay = np.zeros((*mask_slice.shape, 4))
            overlay[mask_slice, 0] = 1.0
            overlay[mask_slice, 3] = 0.7
            axes[1, idx].imshow(overlay, origin='lower')
        axes[1, idx].set_title(f'View {idx+1} - Tumor (RED)')
        axes[1, idx].axis('off')
    
    plt.suptitle(f'Tumor Detection: {os.path.basename(mri_path)}')
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=100)
        print(f"  Saved: {output_path}")
    
    return fig


def main():
    """Main entry point for command-line usage."""
    if len(sys.argv) < 2:
        print("Usage: python detect_tumor.py <path_to_mri.nii.gz> [--threshold]")
        print()
        print("Options:")
        print("  --threshold    Use intensity threshold method instead of U-Net")
        print()
        print("Example:")
        print("  python detect_tumor.py patient_T1pre.nii.gz")
        sys.exit(1)
    
    mri_path = sys.argv[1]
    use_deep_learning = '--threshold' not in sys.argv
    
    if not os.path.exists(mri_path):
        print(f"Error: File not found: {mri_path}")
        sys.exit(1)
    
    # Analyze
    results = analyze_tumor(mri_path, use_deep_learning=use_deep_learning)
    
    # Visualize
    output_path = mri_path.replace('.nii.gz', '_detection.png')
    visualize_result(mri_path, results.get('tumor_mask'), output_path)


if __name__ == '__main__':
    main()