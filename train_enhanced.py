import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from tqdm import tqdm
import numpy as np
from sklearn.metrics import average_precision_score, accuracy_score
import argparse
import wandb
import os
import random
import re
from networks.fake_image_detection_exposing_enhanced import FakeImageDetectionExposingEnhanced
from augment import ImageAugmentor
from mask import FrequencyMaskGenerator, PixelMaskGenerator, PatchMaskGenerator
from utils import train_augment, val_augment
from earlystop import EarlyStopping
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_DEBUG'] = 'WARN'
os.environ['WANDB_CONFIG_DIR'] = './wandb_enhanced'
os.environ['WANDB_DIR'] = './wandb_enhanced'
os.environ['WANDB_CACHE_DIR'] = './wandb_enhanced'

def train_model_enhanced(
    model, 
    train_loader, 
    val_loader, 
    num_epochs=50, 
    resume_epoch=0,
    save_path=None,
    early_stopping=None,
    device=None,
    args=None,
    ):
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    
    best_score = early_stopping.best_score if early_stopping else 0
    counter = early_stopping.counter if early_stopping else 0
    
    train_losses = []
    train_accs = []
    train_aps = []
    val_losses = []
    val_accs = []
    val_aps = []
    
    best_ap = 0.0
    best_acc = 0.0
    best_combined_score = 0.0
    
    for epoch in range(resume_epoch, num_epochs):
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('-' * 10)
        
        model.train()
        running_loss = 0.0
        running_corrects = 0
        total_samples = 0
        all_labels = []
        all_probs = []
        
        pbar = tqdm(train_loader, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images, images)
            
            total_loss, loss_dict = model.compute_loss(outputs, labels)
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
            
            running_loss += total_loss.item()
            _, preds = torch.max(outputs['main_output'], 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            probs = torch.softmax(outputs['main_output'], dim=1)[:, 1]
            all_probs.extend(probs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx == len(train_loader) - 1:
                current_acc = running_corrects.double() / total_samples
                pbar.set_description(f'Training: {100.*current_acc:.2f}%')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_corrects.double() / total_samples
        
        epoch_ap = average_precision_score(all_labels, all_probs) if len(all_labels) > 0 else 0.0
        
        val_loss, val_acc, val_ap = validate_model_enhanced(model, val_loader, device)
        
        train_losses.append(epoch_loss)
        train_accs.append(epoch_acc.cpu().numpy())
        train_aps.append(epoch_ap)
        val_losses.append(val_loss)
        val_accs.append(val_acc.cpu().numpy())
        val_aps.append(val_ap)
        
        print(f'Training Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} AP: {epoch_ap:.4f}')
        print(f'Validation Loss: {val_loss:.4f} Acc: {val_acc:.4f} AP: {val_ap:.4f}')
        print()
        
        if args.wandb_online:
            wandb.log({
                'epoch': epoch + 1,
                'train_loss': epoch_loss,
                'train_acc': epoch_acc,
                'train_ap': epoch_ap,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'val_ap': val_ap,
                'learning_rate': optimizer.param_groups[0]['lr'],
                'fusion_type': 'DDAFM' if args.use_ddafm else 'CrossDomainAttention'
            })
        
        current_combined_score = (val_ap + val_acc) / 2  
        
        if current_combined_score > best_combined_score:
            best_combined_score = current_combined_score
            best_ap = val_ap
            best_acc = val_acc
            
            best_checkpoint_name = f'{save_path}_ddafm_best_combined.pth' if args.use_ddafm else f'{save_path}_best_combined.pth'
            best_checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ap': best_ap,
                'best_acc': best_acc,
                'best_combined_score': best_combined_score,
                'val_loss': val_loss,
                'use_ddafm': args.use_ddafm
            }
            torch.save(best_checkpoint, best_checkpoint_name)
            print(f'New best model saved! AP: {val_ap:.4f}, Acc: {val_acc:.4f}, Combined: {current_combined_score:.4f}, Path: {best_checkpoint_name}')
        
        if early_stopping:
            try:
                early_stopping(val_loss, model, optimizer, epoch)
            except TypeError:
                try:
                    early_stopping(val_loss, model)
                except:
                    pass
            
            if early_stopping.early_stop:
                print("Early stopping")
                break
        
        if (epoch + 1) % 10 == 0:
            checkpoint_name = f'{save_path}_ddafm_epoch_{epoch+1}.pth' if args.use_ddafm else f'{save_path}_epoch_{epoch+1}.pth'
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_ap': best_ap,
                'best_acc': best_acc,
                'best_combined_score': best_combined_score,
                'val_loss': val_loss,
                'use_ddafm': args.use_ddafm
            }
            torch.save(checkpoint, checkpoint_name)
    
    final_checkpoint_name = f'{save_path}_ddafm_final.pth' if args.use_ddafm else f'{save_path}_final.pth'
    final_checkpoint = {
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'best_ap': best_ap,
        'best_acc': best_acc,
        'best_combined_score': best_combined_score,
        'val_loss': val_loss,
        'use_ddafm': args.use_ddafm
    }
    torch.save(final_checkpoint, final_checkpoint_name)
    
    print(f"\nTraining completed!")
    print(f"Fusion Type: {'DDAFM' if args.use_ddafm else 'CrossDomainAttention'}")
    print(f"Best Combined Score: {best_combined_score:.4f}")
    print(f"Best AP: {best_ap:.4f}")
    print(f"Best Accuracy: {best_acc:.4f}")
    print(f"Final Model Path: {final_checkpoint_name}")
    
    return model

def validate_model_enhanced(model, val_loader, device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    total_samples = 0
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, 
                   bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')
        
        for batch_idx, (images, labels) in enumerate(pbar):
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images, images)
            
            total_loss, loss_dict = model.compute_loss(outputs, labels)
            
            running_loss += total_loss.item()
            _, preds = torch.max(outputs['main_output'], 1)
            running_corrects += torch.sum(preds == labels.data)
            total_samples += labels.size(0)
            
            probs = torch.softmax(outputs['main_output'], dim=1)[:, 1]
            all_probs.extend(probs.cpu().detach().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            if batch_idx == len(val_loader) - 1:
                current_acc = running_corrects.double() / total_samples
                pbar.set_description(f'Validation: {100.*current_acc:.2f}%')
    
    epoch_loss = running_loss / len(val_loader) if len(val_loader) > 0 else 0.0
    epoch_acc = running_corrects.double() / total_samples if total_samples > 0 else 0.0
    
    epoch_ap = average_precision_score(all_labels, all_probs) if len(all_labels) > 0 else 0.0
    
    return epoch_loss, epoch_acc, epoch_ap

def main(
    local_rank=0,
    num_epochs=50,
    ratio=15,
    batch_size=32,
    wandb_run_id=None,
    model_name='enhanced',
    band='all',
    wandb_name=None,
    project_name=None,
    save_path=None,
    mask_type='spectral',
    pretrained=False,
    resume_train=None,
    early_stop=True,
    wandb_online=False,
    args=None,
    ):
    seed = 44
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    wandb_resume = "allow" if resume_train else None

    status = "online" if wandb_online else "offline"
    if args.use_ddafm:
        wandb_name = f"ddafm_{wandb_name}"
        project_name = f"DDAFM-{project_name}"
    wandb.init(id=wandb_run_id, resume=wandb_resume, project=project_name, name=wandb_name, mode=status)
    wandb.config.update(args, allow_val_change=True)
    train_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.1,
        'blur_sig': [1.0],
        'jpg_prob': 0.0,
        'jpg_method': ['pil'],
        'jpg_qual': [100]
    }
    val_opt = {
        'rz_interp': ['bilinear'],
        'loadSize': 256,
        'blur_prob': 0.0,
        'blur_sig': [0.0],
        'jpg_prob': 0.0,
        'jpg_method': ['pil'],
        'jpg_qual': [100]
    }
    if ratio > 1.0 or ratio < 0.0:
        raise ValueError(f"Invalid mask ratio {ratio}")
    else:
        if mask_type == 'spectral':
            mask_generator = FrequencyMaskGenerator(ratio=ratio, band=band)
        elif mask_type == 'adaptive_spectral':
            mask_generator = None
        elif mask_type == 'pixel':
            mask_generator = PixelMaskGenerator(ratio=ratio)
        elif mask_type == 'patch':
            mask_generator = PatchMaskGenerator(ratio=ratio)
        else:
            mask_generator = None
    train_transform = train_augment(ImageAugmentor(train_opt), mask_generator, args)
    val_transform = val_augment(ImageAugmentor(val_opt), mask_generator, args)
    train_data = datasets.ImageFolder(root='/root/autodl-tmp/FakeImageDetection/data/Dataset_HI/train', transform=train_transform)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_data = datasets.ImageFolder(root='/root/autodl-tmp/FakeImageDetection/data/Dataset_HI/val', transform=val_transform)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    print(f"Initializing FakeImageDetectionExposingEnhanced model (use_ddafm={args.use_ddafm})...")
    model = FakeImageDetectionExposingEnhanced(
        num_classes=2,
        pretrained=pretrained,
        use_ddafm=args.use_ddafm,
        use_adaptive_mask=(mask_type == 'adaptive_spectral'),
        adaptive_mask_ratio=ratio,
        adaptive_mask_grid=args.adaptive_grid,
        adaptive_mask_hidden_factor=args.adaptive_hidden_factor,
        adaptive_mask_tau=args.adaptive_tau,
    )
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=1e-4)
    last_epoch = -1
    best_ap = 0.0
    best_acc = 0.0
    best_combined_score = 0.0
    counter = 0
    
    if resume_train:
        checkpoint_candidates = []
        if args.use_ddafm:
            checkpoint_candidates.extend([
                f'{save_path}_ddafm_best_combined.pth',
                f'{save_path}_ddafm_final.pth'
            ])
        checkpoint_candidates.extend([
            f'{save_path}_best_combined.pth',
            f'{save_path}_final.pth'
        ])
        
        checkpoint_path = None
        for candidate in checkpoint_candidates:
            if os.path.exists(candidate):
                checkpoint_path = candidate
                break
        
        if checkpoint_path is None:
            folder_path = os.path.dirname(save_path)
            base_filename = os.path.basename(save_path)
            checkpoint_files = os.listdir(folder_path)
            
            if checkpoint_files:
                epoch_files = []
                if args.use_ddafm:
                    epoch_files = [f for f in checkpoint_files if f.startswith(f'{base_filename}_ddafm_epoch_') and f.endswith('.pth')]
                if not epoch_files:
                    epoch_files = [f for f in checkpoint_files if f.startswith(f'{base_filename}_epoch_') and f.endswith('.pth')]
                
                if epoch_files:
                    ep_numbers = []
                    for f in epoch_files:
                        match = re.search(r'_epoch_(\d+)', f)
                        if match:
                            ep_numbers.append(int(match.group(1)))
                    if ep_numbers:
                        max_ep = max(ep_numbers)
                        if args.use_ddafm:
                            checkpoint_path = f'{save_path}_ddafm_epoch_{max_ep}.pth'
                        else:
                            checkpoint_path = f'{save_path}_epoch_{max_ep}.pth'
        
        if checkpoint_path and os.path.exists(checkpoint_path):
            print(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            checkpoint_use_ddafm = checkpoint.get('use_ddafm', False)
            if args.use_ddafm != checkpoint_use_ddafm:
                print(f"Warning: Checkpoint fusion type ({'DDAFM' if checkpoint_use_ddafm else 'CrossDomain'}) does not match current config ({'DDAFM' if args.use_ddafm else 'CrossDomain'})")

            checkpoint_state = checkpoint['model_state_dict']
            model_state = model.state_dict()
            filtered_state = {
                k: v for k, v in checkpoint_state.items()
                if k in model_state and model_state[k].shape == v.shape
            }
            skipped_keys = [k for k in checkpoint_state.keys() if k not in filtered_state]
            if skipped_keys:
                print(f"Skipped {len(skipped_keys)} mismatched keys when resuming training.")
                for key in skipped_keys[:20]:
                    print(f"  skip: {key}")
            model.load_state_dict(filtered_state, strict=False)

            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception as e:
                print(f"Warning: failed to load optimizer state strictly, reinitializing optimizer state. Reason: {e}")
            counter = checkpoint.get('counter', 0)
            last_epoch = checkpoint.get('epoch', -1)
            best_ap = checkpoint.get('best_ap', 0.0)
            best_acc = checkpoint.get('best_acc', 0.0)
            best_combined_score = checkpoint.get('best_combined_score', 0.0)
            print(f"\nResuming training from epoch {last_epoch + 1} using {checkpoint_path}")
            print(f"Previous best - AP: {best_ap:.4f}, Acc: {best_acc:.4f}, Combined: {best_combined_score:.4f}")
            for i, param_group in enumerate(optimizer.param_groups):
                print(f"Resume learning rate: {param_group['lr']}")
        else:
            print("No valid checkpoint files found, starting from scratch")
    if early_stop:
        early_stopping_save_path = f'{save_path}_ddafm' if args.use_ddafm else save_path
        early_stopping = EarlyStopping(
            path=early_stopping_save_path, 
            patience=5, 
            verbose=True, 
            min_lr=args.lr/100,
            early_stopping_enabled=early_stop,
            best_score=best_combined_score,  
            counter=counter,
            args=args
        )
    else:
        early_stopping = None
    resume_epoch = last_epoch + 1 if resume_train and last_epoch != -1 else 0
    trained_model = train_model_enhanced(
        model, 
        train_loader, 
        val_loader, 
        num_epochs=num_epochs, 
        resume_epoch=resume_epoch,
        save_path=save_path,
        early_stopping=early_stopping,
        device=device,
        args=args,
    )
        
    wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeImageDetection with Enhanced Training")
    parser.add_argument('--local_rank', type=int, default=0, help='Local rank for distributed training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs training')
    parser.add_argument(
        '--model_name',
        default='enhanced',
        type=str,
        choices=['enhanced'],
        help='Type of model to use'
    )
    parser.add_argument(
        '--wandb_online', 
        action='store_true', 
        help='Run wandb in online mode'
    )
    parser.add_argument(
        '--project_name', 
        type=str, 
        default="FakeImageDetection-Enhanced",
        help='wandb project name'
    )
    parser.add_argument(
        '--wandb_run_id', 
        type=str, 
        default=None,
        help='wandb run id'
    )
    parser.add_argument(
        '--resume_train', 
        default=None,
        type=str,
        choices=['from_last', 'from_best'],
        help='what epoch to resume training'
    )
    parser.add_argument(
        '--band', 
        default='all',
        type=str,
        choices=['all', 'low', 'mid', 'high'],
        help='Frequency band for spectral masking'
    )
    parser.add_argument(
        '--pretrained', 
        action='store_true', 
        help='For pretraining'
    )
    parser.add_argument(
        '--early_stop', 
        action='store_true', 
        help='For early stopping'
    )
    parser.add_argument(
        '--debug', 
        action='store_true', 
        help='For debugging'
    )
    parser.add_argument(
        '--mask_type', 
        default='spectral', 
        choices=['pixel', 'spectral', 'adaptive_spectral', 'patch', 'nomask'], 
        help='Type of mask generator'
    )
    parser.add_argument(
        '--batch_size', 
        type=int, 
        default=32, 
        help='Batch Size'
    )
    parser.add_argument(
        '--ratio', 
        type=int, 
        default=15, 
        help='Masking ratio'
    )
    parser.add_argument(
        '--lr', 
        type=float, 
        default=0.0001, 
        help='learning rate'
    )
    parser.add_argument(
        '--adaptive_grid',
        type=int,
        default=14,
        help='Grid size used by adaptive spectral masking scorer'
    )
    parser.add_argument(
        '--adaptive_hidden_factor',
        type=int,
        default=1,
        help='Hidden size expansion factor for adaptive spectral masking scorer'
    )
    parser.add_argument(
        '--adaptive_tau',
        type=float,
        default=1.0,
        help='Temperature for adaptive spectral masking soft gate'
    )
    parser.add_argument(
        '--use_ddafm', 
        action='store_true', 
        help='Use Dual Domain Attention Fusion Module (DDAFM) instead of CrossDomainAttentionFusion'
    )
    args = parser.parse_args()
    
    model_name = 'enhanced'
    finetune = 'ft' if args.pretrained else ''
    band = '' if args.band == 'all' else args.band
    if args.mask_type != 'nomask':
        ratio = args.ratio
        ckpt_folder = f'./checkpoints_enhanced/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        if args.use_ddafm:
            save_path = f'{ckpt_folder}/{model_name}{finetune}_{band}{args.mask_type}mask_ddafm'
            wandb_name = f"ddafm_enhanced_mask_{ratio}_{model_name}{finetune}_{band}{args.mask_type}"
        else:
            save_path = f'{ckpt_folder}/{model_name}{finetune}_{band}{args.mask_type}mask'
            wandb_name = f"enhanced_mask_{ratio}_{model_name}{finetune}_{band}{args.mask_type}"
    else:
        ratio = 0
        ckpt_folder = f'./checkpoints_enhanced/mask_{ratio}'
        os.makedirs(ckpt_folder, exist_ok=True)
        if args.use_ddafm:
            save_path = f'{ckpt_folder}/{model_name}{finetune}_ddafm'
            wandb_name = f"ddafm_enhanced_mask_{ratio}_{model_name}{finetune}"
        else:
            save_path = f'{ckpt_folder}/{model_name}{finetune}'
            wandb_name = f"enhanced_mask_{ratio}_{model_name}{finetune}"
    num_epochs = 100 if args.early_stop else args.num_epochs
    print("\nEnhanced FakeImageDetection Configuration:")
    print("-" * 40)
    print(f"Model Type: Enhanced (Pixel + Frequency Domain)")
    print(f"Fusion Module: {'Dual Domain Attention Fusion Module (DDAFM)' if args.use_ddafm else 'Cross Domain Attention Fusion'}")
    print(f"Number of Epochs: {num_epochs}")
    print(f"Early Stopping: {args.early_stop}")
    print(f"Mask Generator Type: {args.mask_type}")
    print(f"Mask Ratio: {ratio}%")
    print(f"Batch Size: {args.batch_size}")
    print(f"Learning Rate: {args.lr}")
    print(f"WandB Project: {args.project_name}")
    print(f"WandB Instance: {wandb_name}")
    print(f"Save path: {save_path}.pth")
    print(f"Resume training: {args.resume_train}")
    print("-" * 40, "\n")
    main(
        local_rank=args.local_rank,
        num_epochs=num_epochs,
        ratio=ratio/100,
        batch_size=args.batch_size,
        wandb_run_id=args.wandb_run_id,
        model_name=args.model_name,
        band=args.band,
        wandb_name=wandb_name,
        project_name=args.project_name,
        save_path=save_path, 
        mask_type=args.mask_type,
        pretrained=args.pretrained,
        resume_train=args.resume_train,
        early_stop=args.early_stop,
        wandb_online=args.wandb_online,
        args=args
    )