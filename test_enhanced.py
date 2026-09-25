import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.metrics import average_precision_score, accuracy_score, roc_auc_score, f1_score
import numpy as np
from PIL import Image
import os
from tqdm import tqdm
import argparse
import random
import torchvision.models as vis_models
from dataset import *
from augment import ImageAugmentor
from mask import *
from utils import *
from networks.fake_image_detection_exposing_enhanced import FakeImageDetectionExposingEnhanced
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'

def evaluate_model_enhanced(model_name, data_type, mask_type, ratio, dataset_path, dataset_label, batch_size, checkpoint_path, device, args):
    
    val_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.0,
        'blur_sig': [0.0],
        'jpg_prob': 0.0,
        'jpg_method': ['pil'],
        'jpg_qual': [100]
    }
    
    if mask_type == 'spectral':
        mask_generator = FrequencyMaskGenerator(ratio=ratio, band=args.band)
    elif mask_type == 'adaptive_spectral':
        mask_generator = None
    elif mask_type == 'pixel':
        mask_generator = PixelMaskGenerator(ratio=ratio)
    elif mask_type == 'patch':
        mask_generator = PatchMaskGenerator(ratio=ratio)
    else:
        mask_generator = None
    
    val_transform = val_augment(ImageAugmentor(val_opt), mask_generator, args)
    
    try:
        test_data = datasets.ImageFolder(root=dataset_path, transform=val_transform)
        test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
        print(f"Successfully loaded {dataset_label} dataset (size: {len(test_data)} samples)")
    except Exception as e:
        print(f"Failed to load {dataset_label} dataset: {str(e)}")
        return dataset_label, 0.0, 0.0, 0.0, 0.0  
    
    if model_name.lower() == 'enhanced':
        model = FakeImageDetectionExposingEnhanced(
            num_classes=2,
            pretrained=False,
            use_ddafm=args.use_ddafm,
            use_adaptive_mask=(mask_type == 'adaptive_spectral'),
            adaptive_mask_ratio=ratio,
            adaptive_mask_grid=args.adaptive_grid,
            adaptive_mask_hidden_factor=args.adaptive_hidden_factor,
            adaptive_mask_tau=args.adaptive_tau,
        )
    else:
        model = create_model(model_name, data_type, mask_type, ratio, device, args)
    
    global model_loaded  
    global loaded_model
    if not model_loaded:
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"\nLoading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
            model_state_dict = checkpoint.get('model_state_dict', checkpoint)
            
            current_model_dict = model.state_dict()
            filtered_state_dict = {k: v for k, v in model_state_dict.items() if k in current_model_dict and current_model_dict[k].shape == v.shape}
            
            for k in model_state_dict.keys():
                if k not in filtered_state_dict:
                    if k in current_model_dict:
                        print(f"Skip {k}: shape mismatch {model_state_dict[k].shape} vs {current_model_dict[k].shape}")
                    else:
                        print(f"Skip {k}: not found in current model")

            model.load_state_dict(filtered_state_dict, strict=False)
            print("Checkpoint loaded with strict=False (some layers may be skipped)")
            loaded_model = model.to(device)
            loaded_model.eval()
            model_loaded = True
        else:
            print(f"Error: Checkpoint not found at {checkpoint_path}")
            return dataset_label, 0.0, 0.0, 0.0, 0.0
    else:
        model = loaded_model  
    
    all_labels = []
    all_probs = []
    all_preds = []
    
    with torch.no_grad():
        pbar = tqdm(test_loader, desc=f'Testing {dataset_label} Dataset')
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            if model_name.lower() == 'enhanced':
                outputs = model(images, images)
                if isinstance(outputs, dict) and 'main_output' in outputs:
                    outputs = outputs['main_output']
            else:
                outputs = model(images)
            
            probs = torch.softmax(outputs, dim=1)[:, 1]
            _, preds = torch.max(outputs, 1)
            
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    ap = average_precision_score(all_labels, all_probs) if len(all_labels) > 0 else 0.0
    acc = accuracy_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    auc = roc_auc_score(all_labels, all_probs) if len(all_labels) > 0 else 0.0
    f1 = f1_score(all_labels, all_preds) if len(all_labels) > 0 else 0.0
    
    print(f"\n{dataset_label} Dataset Evaluation Results:")
    print(f"  Average Precision: {ap:.4f}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  ROC AUC Score: {auc:.4f}")
    print(f"  F1 Score: {f1:.4f}")
    print("-" * 60)
    
    return dataset_label, ap, acc, auc, f1

def create_model(model_name, data_type, mask_type, ratio, device, args):
    if model_name.lower() == 'rn50':
        model = vis_models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name.lower() == 'rn18':
        model = vis_models.resnet18(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    elif model_name.lower() == 'rn34':
        model = vis_models.resnet34(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    else:
        model = vis_models.resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, 2)
    
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test script for FakeImageDetection (Enhanced Model)")
    
    parser.add_argument('--model_name', default='enhanced', type=str, choices=['RN18', 'RN34', 'RN50', 'RN50_mod', 'enhanced'])
    parser.add_argument('--mask_type', default='spectral', choices=['patch', 'spectral', 'adaptive_spectral', 'pixel', 'nomask'])
    parser.add_argument('--band', default='all', type=str, choices=['all', 'low', 'mid', 'high'])
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--ratio', type=int, default=15)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--data_type', default="Wang_CVPR20", type=str, choices=['Wang_CVPR20', 'Ojha_CVPR23'])
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--adaptive_grid', type=int, default=14, help='Grid size used by adaptive spectral masking scorer')
    parser.add_argument('--adaptive_hidden_factor', type=int, default=1, help='Hidden size expansion factor for adaptive spectral masking scorer')
    parser.add_argument('--adaptive_tau', type=float, default=1.0, help='Temperature for adaptive spectral masking soft gate')
    parser.add_argument('--use_ddafm', action='store_true', help='Use Dual Domain Attention Fusion Module checkpoint structure')
    
    parser.add_argument('--checkpoint_file', required=True, type=str)
    args = parser.parse_args()
    
    args.model_name = 'enhanced'
    
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_name = args.model_name.lower()
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band
    ratio = args.ratio
    
    global model_loaded
    global loaded_model
    model_loaded = False
    loaded_model = None
    
    base_dir = '/root/autodl-tmp/FakeImageDetection'
    
    checkpoint_dir = f'{base_dir}/checkpoints_enhanced/mask_{ratio}'
    checkpoint_path = os.path.join(checkpoint_dir, args.checkpoint_file)

    if not os.path.exists(checkpoint_path):
        print(f"Error: 权重文件 {checkpoint_path} 不存在")
        print(f"请确保文件存在于 {checkpoint_dir} 目录下")
        exit(1)
    
    fusion_tag = 'ddafm' if args.use_ddafm else 'crossdomain'
    result_file_path = f'/root/autodl-tmp/FakeImageDetection/results_enhanced/wang_cvpr20/enhanced_{args.mask_type}mask{ratio}_{fusion_tag}_results.txt'
    results_dir = os.path.dirname(result_file_path)
    os.makedirs(results_dir, exist_ok=True)
    
    test_datasets = {
        'SameAsTrain': f'{base_dir}/data/test'
    }
    

    print("\nEnhanced FakeImageDetection Test Configuration:")
    print("-" * 60)
    print(f"Device: {device}")
    print(f"Model Type: {args.model_name}")
    print(f"Fusion Module: {'DDAFM' if args.use_ddafm else 'CrossDomainAttention'}")
    print(f"Mask Type: {args.mask_type}")
    print(f"Mask Ratio: {ratio}%")
    print(f"Batch Size: {args.batch_size}")
    print(f"Checkpoint File: {args.checkpoint_file}")
    print(f"Checkpoint Path: {checkpoint_path}")
    print(f"Total Datasets to Test: {len(test_datasets)}")
    for idx, (label, path) in enumerate(test_datasets.items(), 1):
        print(f"  {idx}. {label}: {path}")
    print(f"Results will be saved to: {result_file_path}")
    print("-" * 60, "\n")
    
    all_results = []
    
    print(f"Starting batch testing of {len(test_datasets)} datasets...\n")
    for dataset_label, dataset_path in test_datasets.items():
        label, ap, acc, auc, f1 = evaluate_model_enhanced(
            args.model_name,
            args.data_type,
            args.mask_type,
            ratio/100,
            dataset_path,
            dataset_label,  
            args.batch_size,
            checkpoint_path,
            device,
            args,
        )
        all_results.append([label, ap, acc, auc, f1])
    
    import time
    import pandas as pd  
    
    with open(result_file_path, 'a') as file:  
        file.write("\n" + "=" * 100 + "\n")
        file.write("Enhanced FakeImageDetection Batch Test Report\n")
        file.write("=" * 100 + "\n")
        file.write(f"Test Time: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        file.write(f"Device: {device}\n")
        file.write(f"Model Type: {args.model_name}\n")
        file.write(f"Mask Type: {args.mask_type}\n")
        file.write(f"Mask Ratio: {ratio}%\n")
        file.write(f"Batch Size: {args.batch_size}\n")
        file.write(f"Checkpoint File: {args.checkpoint_file}\n")
        file.write(f"Checkpoint Path: {checkpoint_path}\n")
        file.write(f"Total Tested Datasets: {len(all_results)}\n")
        file.write("=" * 100 + "\n\n")
        
        file.write(f"{'Dataset Label':<20} {'Average Precision':<20} {'Accuracy':<15} {'ROC AUC':<15} {'F1 Score':<15}\n")
        file.write("-" * 100 + "\n")
        
        for result in all_results:
            label, ap, acc, auc, f1 = result
            file.write(f"{label:<20} {ap:.4f} ({ap*100:.2f}%) {'':<4} {acc:.4f} ({acc*100:.2f}%) {'':<1} {auc:.4f} {'':<6} {f1:.4f} ({f1*100:.2f}%)\n")
        
        file.write("\n" + "-" * 100 + "\n")
        file.write("Summary Statistics:\n")
        valid_results = [r for r in all_results if r[1] > 0]
        if valid_results:
            avg_ap = np.mean([r[1] for r in valid_results])
            avg_acc = np.mean([r[2] for r in valid_results])
            avg_auc = np.mean([r[3] for r in valid_results])
            avg_f1 = np.mean([r[4] for r in valid_results])
            file.write(f"Average of Valid Datasets (n={len(valid_results)}):\n")
            file.write(f"  Average Precision: {avg_ap:.4f} ({avg_ap*100:.2f}%)\n")
            file.write(f"  Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)\n")
            file.write(f"  Average ROC AUC: {avg_auc:.4f}\n")
            file.write(f"  Average F1 Score: {avg_f1:.4f} ({avg_f1*100:.2f}%)\n")
        else:
            file.write("No valid dataset results (all datasets failed to load or evaluate)\n")
        file.write("=" * 100 + "\n\n")
    
    print("\n" + "=" * 80)
    print("Batch Testing Completed! Final Summary Report")
    print("=" * 80)
    print(f"Total Datasets Tested: {len(all_results)}")
    print(f"Valid Datasets (Successfully Evaluated): {len([r for r in all_results if r[1] > 0])}")
    print(f"Failed Datasets (Load/Eval Error): {len([r for r in all_results if r[1] == 0])}")
    print("\nKey Metrics Summary (Valid Datasets):")
    valid_results = [r for r in all_results if r[1] > 0]
    if valid_results:
        avg_ap = np.mean([r[1] for r in valid_results])
        avg_acc = np.mean([r[2] for r in valid_results])
        avg_auc = np.mean([r[3] for r in valid_results])
        avg_f1 = np.mean([r[4] for r in valid_results])
        print(f"  Average Precision: {avg_ap:.4f} ({avg_ap*100:.2f}%)")
        print(f"  Average Accuracy: {avg_acc:.4f} ({avg_acc*100:.2f}%)")
        print(f"  Average ROC AUC: {avg_auc:.4f}")
        print(f"  Average F1 Score: {avg_f1:.4f} ({avg_f1*100:.2f}%)")
    else:
        print("  No valid dataset results available.")
    print(f"\nDetailed results appended to: {result_file_path}")
    print("=" * 80)