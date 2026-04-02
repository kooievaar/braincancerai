#!/usr/bin/env python3
"""
Evaluation script for Brain Tumor Detection Model
Evaluates model performance on test dataset with detailed metrics.

Usage:
    python evaluate_test.py                 # Evaluate on test set
    python evaluate_test.py --dataset train  # Evaluate on training set
    python evaluate_test.py --detailed      # Show detailed metrics
    python evaluate_test.py --compare       # Compare with threshold method
"""

import nibabel as nib
import numpy as np
import os
import json
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from scipy.ndimage import label

# Configuration
DETECTION_THRESHOLD = 244.7
TEST_PATH = 'mri-scans/UCSD-PTGBM-BraTS-2024-test-set MRI-data 2'


def load_mri(path):
    """Load MRI file."""
    img = nib.load(path)
    return img.get_fdata(), img.affine, img.header


def detect_tumor_threshold(mri_data, threshold=DETECTION_THRESHOLD):
    """
    Threshold-based tumor detection.
    Note: Less accurate than U-Net deep learning approach.
    """
    data_max = mri_data.max()
    norm_threshold = threshold / data_max if data_max > 0 else 0.15
    
    tumor_mask = mri_data > norm_threshold
    
    # Filter small regions
    labeled, num = label(tumor_mask)
    filtered = np.zeros_like(tumor_mask)
    for i in range(1, num + 1):
        if np.sum(labeled == i) > 100:
            filtered[labeled == i] = True
    
    return filtered


def evaluate_patient(mri_path, seg_path):
    """
    Evaluate detection for a single patient.
    
    Returns dict with metrics.
    """
    # Load MRI and ground truth
    mri_data, _, _ = load_mri(mri_path)
    seg_data, _, _ = load_mri(seg_path)
    
    # Ground truth: any non-zero in segmentation is tumor
    ground_truth = (seg_data > 0).astype(int)
    
    # Detection using threshold method
    # Note: For production, use trained U-Net model
    prediction = detect_tumor_threshold(mri_data).astype(int)
    
    # Calculate metrics
    y_true = ground_truth.flatten()
    y_pred = prediction.flatten()
    
    # Overall metrics
    try:
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
    except:
        precision = recall = f1 = 0
    
    # Dice score
    intersection = np.sum(y_true * y_pred)
    dice = (2. * intersection + 1e-6) / (np.sum(y_true) + np.sum(y_pred) + 1e-6)
    
    # IoU (Jaccard)
    union = np.sum(y_true) + np.sum(y_pred) - intersection
    iou = (intersection + 1e-6) / (union + 1e-6)
    
    # Patient-level metrics
    actual_has_tumor = np.sum(ground_truth) > 0
    predicted_has_tumor = np.sum(prediction) > 100  # At least 100 voxels
    
    return {
        'dice': float(dice),
        'iou': float(iou),
        'precision': float(precision),
        'recall': float(recall),
        'f1': float(f1),
        'actual_tumor_voxels': int(np.sum(ground_truth)),
        'predicted_tumor_voxels': int(np.sum(prediction)),
        'actual_has_tumor': actual_has_tumor,
        'predicted_has_tumor': predicted_has_tumor,
    }


def evaluate_dataset(data_path, dataset_name):
    """Evaluate all patients in a dataset."""
    print(f"\nEvaluating {dataset_name}...")
    
    patients = sorted([d for d in os.listdir(data_path) if d.startswith('UCSD-PTGBM')])
    
    results = []
    for i, patient in enumerate(patients):
        mri_file = os.path.join(data_path, patient, patient + '_T1pre.nii.gz')
        seg_file = os.path.join(data_path, patient, patient + '_BraTS_tumor_seg.nii.gz')
        
        if os.path.exists(mri_file) and os.path.exists(seg_file):
            try:
                result = evaluate_patient(mri_file, seg_file)
                result['patient'] = patient
                results.append(result)
                
                if (i+1) % 20 == 0:
                    print(f"  Progress: {i+1}/{len(patients)}")
            except Exception as e:
                print(f"  Error {patient}: {e}")
    
    return results


def calculate_summary(results):
    """Calculate aggregate metrics."""
    if not results:
        return {}
    
    # Per-patient metrics (tumor/no-tumor classification)
    tp = sum(1 for r in results if r['actual_has_tumor'] and r['predicted_has_tumor'])
    fp = sum(1 for r in results if not r['actual_has_tumor'] and r['predicted_has_tumor'])
    tn = sum(1 for r in results if not r['actual_has_tumor'] and not r['predicted_has_tumor'])
    fn = sum(1 for r in results if r['actual_has_tumor'] and not r['predicted_has_tumor'])
    
    # Voxel-level metrics
    avg_dice = np.mean([r['dice'] for r in results])
    avg_iou = np.mean([r['iou'] for r in results])
    avg_precision = np.mean([r['precision'] for r in results])
    avg_recall = np.mean([r['recall'] for r in results])
    
    # Patient-level accuracy
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # False positive rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0  # False negative rate
    
    return {
        'num_patients': len(results),
        'patient_level': {
            'accuracy': accuracy,
            'fpr': fpr,
            'fnr': fnr,
            'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        },
        'voxel_level': {
            'avg_dice': avg_dice,
            'avg_iou': avg_iou,
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
        }
    }


def print_results(summary, dataset_name):
    """Print evaluation results."""
    print(f"\n{'='*60}")
    print(f"RESULTS: {dataset_name}")
    print(f"{'='*60}")
    
    print(f"\nPatient-Level Metrics ({summary['num_patients']} patients):")
    pl = summary['patient_level']
    print(f"  Accuracy: {pl['accuracy']*100:.2f}%")
    print(f"  False Positive Rate: {pl['fpr']*100:.2f}%")
    print(f"  False Negative Rate: {pl['fnr']*100:.2f}%")
    print(f"  TP: {pl['tp']}, FP: {pl['fp']}, TN: {pl['tn']}, FN: {pl['fn']}")
    
    print(f"\nVoxel-Level Metrics:")
    vl = summary['voxel_level']
    print(f"  Average Dice Score: {vl['avg_dice']:.4f}")
    print(f"  Average IoU (Jaccard): {vl['avg_iou']:.4f}")
    print(f"  Average Precision: {vl['avg_precision']:.4f}")
    print(f"  Average Recall: {vl['avg_recall']:.4f}")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate brain tumor detection')
    parser.add_argument('--dataset', default='test', choices=['train', 'test', 'all'],
                        help='Which dataset to evaluate')
    parser.add_argument('--detailed', action='store_true', help='Show detailed per-patient results')
    parser.add_argument('--output', type=str, help='Output JSON file for results')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("BRAIN TUMOR DETECTION EVALUATION")
    print("=" * 60)
    
    if args.dataset in ['test', 'all']:
        test_results = evaluate_dataset(TEST_PATH, "Test Set")
        test_summary = calculate_summary(test_results)
        print_results(test_summary, "Test Set")
    
    if args.dataset in ['train', 'all']:
        train_path = 'mri-scans/UCSD-PTGBM training MRI-data 1'
        if os.path.exists(train_path):
            train_results = evaluate_dataset(train_path, "Training Set")
            train_summary = calculate_summary(train_results)
            print_results(train_summary, "Training Set")
    
    # Save results if requested
    if args.output:
        output_data = {
            'test_summary': test_summary if args.dataset in ['test', 'all'] else None,
        }
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()