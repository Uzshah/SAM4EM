from __future__ import absolute_import, division, print_function

import os
import numpy as np
import time
import json
import tqdm
import random
from collections import OrderedDict
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from dataset_loader import EM_dataset, RandomGenerator
from metric import Evaluator
from torch.optim import lr_scheduler
from sam2.build_sam import build_sam
from LoRa_image_encoder2 import LoRA_Sam as lora_sam2
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from matplotlib import pyplot as plt
import math
from lr_schedular import WarmupCosineLR
from transformers import AdamW


# Learning rate scheduler with minimum LR cap
class StepLRWithMinLR(optim.lr_scheduler.StepLR):
    def __init__(self, optimizer, step_size, gamma=0.1, min_lr=1e-6, last_epoch=-1):
        self.min_lr = min_lr
        super().__init__(optimizer, step_size, gamma, last_epoch)

    def step(self, epoch=None):
        current_lr = self.optimizer.param_groups[0]['lr']
        if current_lr > self.min_lr:
            super().step(epoch)
        
        for param_group in self.optimizer.param_groups:
            if param_group['lr'] < self.min_lr:
                param_group['lr'] = self.min_lr


# Cosine annealing with warmup
class CosineAnnealingWithWarmup:
    def __init__(
        self, 
        optimizer: optim.Optimizer,
        warmup_epochs: int,
        total_epochs: int,
        min_lr: float = 1e-7,
        warmup_start_lr: float = 1e-6
    ):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.warmup_start_lr = warmup_start_lr
        self.last_epoch = -1
        
        # Get base learning rate from optimizer
        self.base_lr = optimizer.param_groups[0]['lr']
        
        # Validate parameters
        if warmup_epochs < 0:
            raise ValueError("Warmup epochs must be non-negative")
        if total_epochs <= warmup_epochs:
            raise ValueError("Total epochs must be greater than warmup epochs")
        if min_lr < 0:
            raise ValueError("Minimum learning rate must be non-negative")
        if warmup_start_lr < 0:
            raise ValueError("Warmup start learning rate must be non-negative")
            
    def step(self, epoch: int):
        self.last_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Linear warmup
            progress = epoch / self.warmup_epochs
            lr = self.warmup_start_lr + progress * (self.base_lr - self.warmup_start_lr)
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # Ensure progress doesn't exceed 1
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_decay
        
        # Update learning rate for all parameter groups
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = max(lr, self.min_lr)
    
    def get_last_lr(self) -> list:
        return [group['lr'] for group in self.optimizer.param_groups]
        
    def state_dict(self) -> dict:
        """Returns the state of the scheduler as a dict."""
        return {
            'base_lr': self.base_lr,
            'last_epoch': self.last_epoch,
            'warmup_epochs': self.warmup_epochs,
            'total_epochs': self.total_epochs,
            'min_lr': self.min_lr,
            'warmup_start_lr': self.warmup_start_lr
        }
        
    def load_state_dict(self, state_dict: dict) -> None:
        """Loads the scheduler state."""
        self.base_lr = state_dict['base_lr']
        self.last_epoch = state_dict['last_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.total_epochs = state_dict['total_epochs']
        self.min_lr = state_dict['min_lr']
        self.warmup_start_lr = state_dict['warmup_start_lr']


# Loss calculation
def calc_loss(outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight:float=0.8):
    low_res_logits = outputs['low_res_logits']
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = ((1 - dice_weight) * loss_ce + dice_weight * loss_dice)
    return loss, loss_dice, loss_ce


# IOU Loss implementation
class IOULoss(nn.Module):
    def __init__(self):
        super(IOULoss, self).__init__()
        
    def forward(self, pred, target, is_sigmoid=True):
        if is_sigmoid:
            pred = torch.sigmoid(pred)
        pred = pred.float()
        target = target.float()
        intersection = torch.sum(pred * target)
        union = torch.sum(pred) + torch.sum(target) - intersection
        iou = 1 - (intersection / (union + 1))
        return iou


# Dice Loss implementation        
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, preds, targets):
        # Apply sigmoid to get probabilities if needed
        preds = torch.sigmoid(preds)
        
        # Flatten tensors to calculate intersection and union
        preds = preds.view(-1)
        targets = targets.view(-1)
        
        # Calculate intersection and union
        intersection = (preds * targets).sum()
        dice = (2. * intersection + self.smooth) / (preds.sum() + targets.sum() + self.smooth)
        
        # Return Dice Loss
        return 1 - dice


# Distributed training setup
def setup_for_distributed(is_master):
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(args):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        args.rank = int(os.environ["RANK"])
        args.world_size = int(os.environ['WORLD_SIZE'])
        args.gpu = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        args.rank = int(os.environ['SLURM_PROCID'])
        args.gpu = args.rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        args.distributed = False
        return

    args.distributed = True

    torch.cuda.set_device(args.gpu)
    args.dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(
        args.rank, args.dist_url), flush=True)
    torch.distributed.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                         world_size=args.world_size, rank=args.rank)
    torch.distributed.barrier()
    setup_for_distributed(args.rank == 0)


# Set random seeds for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


class Trainer_sam4em:
    def __init__(self, args):
        init_distributed_mode(args)
        self.args = args
        print(self.args)
        
        # Initialize paths and device
        self.log_path = os.path.join(self.args.log_dir, 
                                     self.args.model_name+"_"+self.args.img_encoder+"_"+self.args.dataset)
        self.device = torch.device(args.device)

        # Initialize datasets
        self._init_datasets()
        
        # Set random seed
        set_seed(args.seed)
        
        # Initialize model and optimizer
        self._init_model()
        self._init_losses_and_metrics()
        
        # Initialize learning rate scheduler
        self._init_lr_scheduler()
        
        # Load weights if specified
        if args.load_weights_dir is not None:
            self.load_model()
            
        # Save experiment settings
        self.save_settings()
    
    def _init_datasets(self):
        """Initialize training and testing datasets and dataloaders"""
        common_transform = transforms.Compose([
            RandomGenerator(
                output_size=[self.args.img_size, self.args.img_size], 
                low_res=[self.args.img_size//4, self.args.img_size//4], 
                norm_type="balanced"
            )
        ])
        
        test_transform = transforms.Compose([
            RandomGenerator(
                output_size=[self.args.img_size, self.args.img_size], 
                low_res=[self.args.img_size//4, self.args.img_size//4], 
                mod=False, 
                norm_type="balanced"
            )
        ])
        
        # Create training dataset
        db_train = EM_dataset(
            base_dir=self.args.root_path, 
            img_dir=self.args.input_dir, 
            label_dir=self.args.target_dir,
            img_size=self.args.img_size,
            transform=common_transform, 
            list_folders=self.args.train_folders, 
            mask_value=self.args.mask_value
        )

        # Create testing dataset
        db_test = EM_dataset(
            base_dir=self.args.root_path, 
            img_dir=self.args.input_dir, 
            label_dir=self.args.target_dir,
            img_size=self.args.img_size,
            transform=test_transform, 
            list_folders=self.args.test_folders, 
            is_training=False, 
            mask_value=self.args.mask_value
        )
        
        # Create dataloaders
        self.trainloader = DataLoader(
            db_train, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True
        )
        
        self.testloader = DataLoader(
            db_test, 
            batch_size=self.args.batch_size, 
            shuffle=False, 
            num_workers=8, 
            pin_memory=True
        )
    
    def _init_model(self):
        """Initialize the model architecture"""
        # Load pretrained model
        ckpt_sam2 = "./pretrained_weights/sam2_hiera_base_plus.pt"
        
        sam = build_sam(
            checkpoint_sam2=ckpt_sam2, 
            is_double_masking=self.args.double_mask,
            img_size=self.args.img_size, 
            num_classes=1,
            mem_slot=self.args.mem_slot, 
            mem_avg_parm=self.args.mem_avg_parm
        )

        # Apply LoRA adaptation
        self.model = lora_sam2(sam, self.args.lora)
        self.model.to(self.device)
        
        # Enable distributed training if needed
        if self.args.distributed:
            self.model = DDP(self.model, device_ids=[self.args.device], find_unused_parameters=True)

        # Print model statistics
        model_total_params = sum(p.numel() for p in self.model.parameters())
        model_grad_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print('model_grad_params:' + str(model_grad_params), '\nmodel_total_params:' + str(model_total_params))
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()), 
            lr=self.args.base_lr
        )
    
    def _init_losses_and_metrics(self):
        """Initialize loss functions and metrics"""
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.ce_loss = nn.CrossEntropyLoss()
        self.dice_loss = DiceLoss()
        self.miou = IOULoss()
        self.mse_loss = nn.MSELoss()
        self.evaluator = Evaluator()
        
    def _init_lr_scheduler(self):
        """Initialize learning rate scheduler with cosine annealing and warmup"""
        if self.args.warmup:
            # Use custom CosineAnnealingWithWarmup for epoch-based scheduling
            self.scheduler = CosineAnnealingWithWarmup(
                optimizer=self.optimizer,
                warmup_epochs=int(self.args.warmup_period / len(self.trainloader)),  # Convert iterations to epochs
                total_epochs=self.args.num_epochs,
                min_lr=1e-6,
                warmup_start_lr=self.args.base_lr * 0.01  # Start with 1% of base_lr
            )
        else:
            # Use standard cosine annealing
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, 
                T_max=self.args.num_epochs,
                eta_min=1e-6
            )
    
    def rgb_to_gray(self, rgb_image):
        rgb_image = rgb_image.transpose((1, 2, 0))
        gray_image = 0.2989 * rgb_image[:, :, 0] + 0.5870 * rgb_image[:, :, 1] + 0.1140 * rgb_image[:, :, 2]
        return gray_image
        
    def imshow(self, img, ax, title):
        npimg = img.cpu().detach().numpy()
        if npimg.shape[0]==3:
            npimg = self.rgb_to_gray(npimg)
        ax.imshow(npimg, cmap='gray')
        ax.set_title(title)
        ax.axis('off')
    
    def vis(self):
        """Visualize model predictions on random batch from test set"""
        self.model.eval()
        output_path = os.path.join(self.log_path, "output_image")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        
        # Create figure for visualization
        fig, axes = plt.subplots(self.args.batch_size, 3, figsize=(15, 25))
        
        # Select a random batch
        total_batches = len(self.testloader)
        random_batch_idx = torch.randint(0, total_batches, (1,)).item()
        
        for i, batch_input in enumerate(self.testloader):
            if i != random_batch_idx:
                continue
                
            # Process the batch
            for key, ipt in batch_input.items():
                if key != "case_name":
                    batch_input[key] = ipt.to(self.device)
                    
            with torch.no_grad():   
                outputs, _ = self.model(
                    batch_input['image'], 
                    previous_mask=batch_input['label'],
                    memory_encoded=None
                )
            
            # Process outputs
            outputs = torch.sigmoid(outputs)
            outputs = (outputs > self.args.threshold).float() 
            
            # Show all images in the batch
            for j in range(self.args.batch_size):
                self.imshow(batch_input['image'][j], axes[j, 0], f'Original {j+1}')
                self.imshow(outputs[j].squeeze(0), axes[j, 1], f'Predict {j+1}')
                self.imshow(batch_input['label'][j], axes[j, 2], f'Ground truth {j+1}')
            
            break
        
        # Add title and save
        plt.suptitle(f'Visualization for Random Batch (Batch {random_batch_idx})', y=1.02, fontsize=16)
        plt.tight_layout()
        plt.savefig(f'{output_path}/{self.epoch}_output_image.png')
        plt.close()
    
    def process_batch(self, inputs, pre_mask):
        """Process a batch of inputs and generate predictions"""
        # Move inputs to device
        for key, ipt in inputs.items():
            if key != "case_name":
                inputs[key] = ipt.to(self.device)
        
        # Set mask for memory-based prediction
        if pre_mask is None:
            mask = inputs['label'][0].unsqueeze(0)
        else:
            mask = pre_mask[-1]
            
        # Generate predictions
        pred_mask, pred_hint = self.model(inputs['image'], previous_mask=mask)
        
        # Calculate loss
        loss = (1-self.args.dice_parm)*self.bce_loss(pred_mask, inputs['label'].unsqueeze(1).float()) + \
                self.args.dice_parm*self.dice_loss(pred_mask, inputs['label'].float())
        
        # Handle double mask if enabled
        if self.args.double_mask:
            loss2 = (1-self.args.dice_parm)*self.bce_loss(pred_hint, inputs['label'].unsqueeze(1).float()) + \
                self.args.dice_parm*self.dice_loss(pred_hint, inputs['label'].float())
            loss = self.args.mask1_parm*loss + (1-self.args.mask1_parm)*loss2
            
        # Use combined masks if specified
        if self.args.use_two_mask:
            mask = 0.5*pred_mask + 0.5*pred_hint
        else:
            mask = pred_mask
            
        return mask, loss
    
    def train(self):
        """Run the entire training pipeline"""
        self.epoch = 0
        self.step = 0
        self.start_time = time.time()
        
        # Validate initially to get baseline performance
        best_performance, _ = self.validate()
        best_epoch = self.args.start_epoch
        
        # Main training loop
        for self.epoch in range(self.args.start_epoch, self.args.num_epochs):
            self.train_one_epoch()
            
            # Validate after each epoch
            miou, val_loss = self.validate()
            
            # Save model after each epoch
            self.save_model(is_last=True)
            
            # Update best performance
            if miou >= best_performance:
                best_performance = miou
                best_epoch = self.epoch
                self.evaluator.print(os.path.join(self.log_path, "best_training.txt"))
                self.save_model(is_ema=False)
                print(f"{best_performance:.4f} best iou at epoch {best_epoch}")
            else:
                print(f"So far {best_performance:.4f} is best iou and best epoch is {best_epoch}")
    
    def train_one_epoch(self):
        """Run a single epoch of training"""
        self.evaluator.reset_eval_metrics()
        self.model.train()
        self.model.sam.mem.memory.zero_()
        
        # Setup progress bar
        pbar = tqdm.tqdm(self.trainloader)
        pbar.set_description(f"Training Epoch_{self.epoch}")
        training_loss = 0
        outputs = None
        
        # Iterate through batches
        for batch_idx, inputs in enumerate(pbar):
            # Process batch and compute loss
            outputs, loss = self.process_batch(inputs, outputs)
            training_loss += loss.item()
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            self.optimizer.zero_grad()
            
            # Current learning rate for display
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Update progress bar
            pbar.set_postfix({
                'TL': training_loss/(1+batch_idx),
                'LR': f"{current_lr:.2e}"
            }, refresh=True)
            
            # Compute metrics for tracking
            with torch.no_grad():
                outputs = torch.sigmoid(outputs)
                outputs = (outputs > self.args.threshold).float()
                self.evaluator.compute_eval_metrics(inputs['label'].detach().cpu(), outputs.detach().cpu())
        
        # Update epoch-based scheduler
        self.scheduler.step(self.epoch)
            
        # Print metrics
        mIoU, Acc, Dice, mAP, _, _, _ = self.evaluator.return_metrics()
        print("Training progress...")
        self.evaluator.print()
    
    def validate(self):
        """Validate the model on the validation set"""
        self.model.eval()
        self.evaluator.reset_eval_metrics()
        self.model.sam.mem.memory.zero_()
        
        # Setup progress bar
        pbar = tqdm.tqdm(self.testloader)
        pbar.set_description(f"Validating Epoch_{self.epoch}")
        v_loss = 0
        outputs = None
        
        with torch.no_grad():
            with torch.cuda.amp.autocast():
                for batch_idx, inputs in enumerate(pbar):
                    # Process batch
                    outputs, loss = self.process_batch(inputs, outputs)
                    v_loss += loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({'VL': v_loss/(1+batch_idx)}, refresh=True)
                    
                    # Compute metrics
                    outputs = torch.sigmoid(outputs)
                    outputs = (outputs > self.args.threshold).float()
                    self.evaluator.compute_eval_metrics(inputs['label'].detach().cpu(), outputs.detach().cpu())
                    
        # Get and print metrics
        mIoU, Acc, Dice, mAP, _, _, _ = self.evaluator.return_metrics()
        print("Validation progress...")
        self.evaluator.print(os.path.join(self.log_path, "training.txt"))
        return mIoU, v_loss/len(self.testloader)
    
    def save_model(self, is_last=False, is_ema=False):
        """Save model weights and optimizer state to disk"""
        save_folder = os.path.join(self.log_path, "models", "best_weights")
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
            
        # Determine save path
        if is_last:
            save_path = os.path.join(save_folder, "last.pth")
        else:
            save_path = os.path.join(save_folder, "model.pth")
            
        # Create state dict
        to_save = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch
        }
        
        # Add scheduler state dict if available
        if hasattr(self.scheduler, 'state_dict'):
            to_save['scheduler_state_dict'] = self.scheduler.state_dict()
        
        # Save to disk
        torch.save(to_save, save_path)
    
    def load_model(self):
        """Load model from disk"""
        self.args.load_weights_dir = os.path.expanduser(self.args.load_weights_dir)
        assert os.path.isdir(self.args.load_weights_dir), \
            f"Cannot find folder {self.args.load_weights_dir}"
        
        print(f"Loading model from folder {self.args.load_weights_dir}")
        
        # Load checkpoint
        path = os.path.join(self.args.load_weights_dir, "model.pth")
        checkpoint = torch.load(path, map_location="cpu")
        
        # Load model weights
        model_dict = checkpoint['model_state_dict']
        if not self.args.distributed:
            model_dict = OrderedDict((key.replace('module.', '', 1) 
                                    if key.startswith('module.') else key, value)
                                   for key, value in model_dict.items())
                
        self.model.load_state_dict(model_dict)
        
        # Load optimizer state if available
        if 'optimizer_state_dict' in checkpoint:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
        # Load scheduler state if available
        if 'scheduler_state_dict' in checkpoint and hasattr(self.scheduler, 'load_state_dict'):
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                print("Loaded scheduler state from checkpoint")
            except Exception as e:
                print(f"Warning: Failed to load scheduler state. Starting with fresh scheduler. Error: {e}")
            
        # Load epoch if available
        if 'epoch' in checkpoint:
            self.epoch = checkpoint['epoch']
            print(f"Resuming from epoch {self.epoch}")
    
    def visualize_and_save_predictions(self, epoch, outputs, inputs, save_dir):
        """Save prediction masks as images"""
        import cv2
        
        # Create directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Convert predictions to numpy arrays
        predictions = outputs.detach().cpu().numpy()
        case_names = inputs['case_name']
        
        # Save each prediction
        for idx in range(predictions.shape[0]):
            # Convert prediction to uint8 format
            pred_mask = (predictions[idx, 0] * 255).astype(np.uint8)
            
            # Save with case name
            save_path = os.path.join(save_dir, f'{self.args.dataset}_{case_names[idx][:-4]}.png')
            cv2.imwrite(save_path, pred_mask)
    
    def save_settings(self):
        """Save settings to disk"""
        models_dir = os.path.join(self.log_path, "models")
        if not os.path.exists(models_dir):
            os.makedirs(models_dir)
            
        # Convert args to dict and save as JSON
        to_save = self.args.__dict__.copy()
        with open(os.path.join(models_dir, 'settings.json'), 'w') as f:
            json.dump(to_save, f, indent=2)