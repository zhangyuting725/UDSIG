#!/usr/bin/env python3
"""
Training script for preference models 
Train the preference model before preference reinforcement learning

Supported model types:
    - DIN: Single-layer attention mechanism CTR prediction model
    - SASRec: Transformer-based sequence CTR prediction model
"""
import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import data_utils
from models.preference_model import *


data_utils.setup_logger()
logger = logging.getLogger(__name__)

class PrefDataset(Dataset):
    """
    DIN preference model training dataset
    Build positive and negative samples based on user historical interactions and click behavior
    """
    
    def __init__(self, args, train_data, image_features, max_seq_len=15):
        self.args = args
        self.image_features = image_features
        self.max_seq_len = max_seq_len
        self.item_num = args.item_num
        # Build training samples
        self.samples = self._build_samples(train_data)
        logger.info(f"Built {len(self.samples)} training samples")
        
    def _build_samples(self, train_data):
        """Build training samples: positive samples (user clicked items) + negative samples (random sampling)"""
        samples = []
        
        # Get all item IDs
        all_item_ids = set(range(self.args.item_num))
        
        
        uids = train_data['uids']
        target_iids = train_data['iids']
        his_iids_list = train_data['his_iids']
        
        # Build user to interacted items mapping
        user_to_items = {}
        for i, uid in enumerate(uids):
            if uid not in user_to_items:
                user_to_items[uid] = []
            user_to_items[uid].append(target_iids[i])
        
        # Build training data for each sample
        for i in range(len(uids)):
            uid = uids[i]
            target_item = target_iids[i]
            history_items = his_iids_list[i]
            
            # Process history sequence: truncate or pad
            if isinstance(history_items, list):
                history_items = history_items.copy()
            else:
                # If tensor, convert to list
                history_items = history_items.tolist() if hasattr(history_items, 'tolist') else list(history_items)
            
            
            # Pad with self.item_num
            if len(history_items) > self.max_seq_len:
                history_items = history_items[-self.max_seq_len:]
            else: 
                history_items = [self.item_num] * (self.max_seq_len - len(history_items)) + history_items
            
            # Positive sample: items actually clicked by user
            samples.append({
                'user_id': uid,
                'history_items': history_items,
                'target_item': target_item,
                'label': 1.0  # Positive sample
            })
            
            # Negative sample: randomly sample items not interacted by user
            user_item_set = set(user_to_items[uid])
            negative_candidates = list(all_item_ids - user_item_set)
            
           
            # Randomly select negative sample
            neg_target = np.random.choice(negative_candidates)
            samples.append({
                'user_id': uid,
                'history_items': history_items,
                'target_item': neg_target,
                'label': 0.0  # Negative sample
            })
            
        return samples
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Get image features of historical items
        history_items = torch.tensor(sample['history_items'], dtype=torch.long)
        
        # Ensure index tensor is on the same device as image_features
        device = self.image_features.device
        history_items = history_items.to(device)
        
        # Create mask (self.item_num indicates padding)
        mask = (history_items != self.item_num).float()  # (max_seq_len,)
        
        # Safely get historical features: filter padding values to avoid index overflow
        valid_mask = history_items != self.item_num
        history_features = torch.zeros(self.max_seq_len, self.image_features.size(1), 
                                     device=device, dtype=self.image_features.dtype)
        
        if valid_mask.any():
            # Only index valid positions
            valid_indices = history_items[valid_mask]
            history_features[valid_mask] = self.image_features[valid_indices]
        
        # Regular mode: return target item and label
        target_item = torch.tensor(sample['target_item'], dtype=torch.long).to(device)
        target_features = self.image_features[target_item]  # (feature_dim,)
        
        return {
            'user_id': sample['user_id'],
            'history_items': sample['history_items'],
            'history_features': history_features,  # (max_seq_len, feature_dim)
            'target_features': target_features,    # (feature_dim,)
            'mask': mask.to(device),               # (max_seq_len,)
            'label': torch.tensor(sample['label'], dtype=torch.float32, device=device)  # scalar
        }





class PreferenceTrainer:
    """Preference model trainer (supports DIN and SASRec)"""
    
    def __init__(self, model, args, device="cuda", model_type='DIN'):
        self.model = model.to(device)
        self.args = args
        self.device = device
        self.model_type = model_type  # 'DIN' or 'SASRec'
        
        # Optimizer - use smaller learning rate for SASRec
        if model_type == 'SASRec':
            lr = args.lr * 0.5  # SASRec uses half learning rate
        else:
            lr = args.lr
            
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=lr,
            weight_decay=args.wd,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate scheduler - add warmup
        self.warmup_epochs = 2
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=args.epochs - self.warmup_epochs
        )
        
        # Loss function
        self.criterion = nn.BCEWithLogitsLoss()  # Binary classification loss
        
        # Training statistics
        self.train_stats = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': []
        }
        
    def train_epoch(self, train_loader):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0.0
        
        for batch_idx, batch in enumerate(train_loader):
            # Move data to device
            history_features = batch['history_features'].to(self.device)  # (batch, seq, feat)
            mask = batch['mask'].to(self.device)                          # (batch, seq)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Regular mode: binary classification
            target_features = batch['target_features'].to(self.device)    # (batch, feat)
            labels = batch['label'].to(self.device)                       # (batch,)
            
            predictions, _ = self.model(
                history_features, target_features, mask
            )
            predictions = predictions.squeeze()  # (batch,)
            
            # Calculate loss
            loss = self.criterion(predictions, labels)
            
            # Backward pass
            loss.backward()
            # Gradient clipping to avoid gradient explosion
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)
            self.optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                logger.info(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        
        return avg_loss
    
    def validate(self, val_loader):
        """Validate model"""
        self.model.eval()
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in val_loader:
                history_features = batch['history_features'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                # Regular mode: binary classification
                target_features = batch['target_features'].to(self.device)
                labels = batch['label'].to(self.device)
                
                # Get logits from model 
                self.model.train()
                logits, _ = self.model(history_features, target_features, mask)
                self.model.eval()
                logits = logits.squeeze()
                
                # Calculate loss (using logits)
                loss = self.criterion(logits, labels)
                
                # Get probabilities for evaluation
                predictions = torch.sigmoid(logits)
                
                # Statistics
                pred_np = predictions.cpu().numpy()
                label_np = labels.cpu().numpy()
                all_predictions.extend(pred_np)
                all_labels.extend(label_np)
                
                total_loss += loss.item()
        
        # Calculate metrics
        avg_loss = total_loss / len(val_loader)
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        auc = roc_auc_score(all_labels, all_predictions)
        
        return avg_loss, auc
    
    def train(self, train_loader, val_loader, save_path=None):
        """Complete training process"""
        if self.model_type == 'SASRec':
            model_name = "SASRec"
        else:
            model_name = "DIN"
        
        logger.info(f"Start training {model_name} preference model, total {self.args.epochs} epochs")
        
        best_auc = 0.0
        
        for epoch in range(self.args.epochs):
            # Warmup learning rate
            if epoch < self.warmup_epochs:
                warmup_factor = (epoch + 1) / self.warmup_epochs
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = self.optimizer.defaults['lr'] * warmup_factor
                logger.info(f"Epoch {epoch+1}/{self.args.epochs} (Warmup: {warmup_factor:.2f})")
            else:
                logger.info(f"Epoch {epoch+1}/{self.args.epochs}")
            
            # Training
            train_loss = self.train_epoch(train_loader)
            
            # Validation
            val_loss, val_auc = self.validate(val_loader)
            
            # Update learning rate (use scheduler after warmup)
            if epoch >= self.warmup_epochs:
                self.scheduler.step()
            
            # Record statistics
            self.train_stats['train_loss'].append(train_loss)
            self.train_stats['val_loss'].append(val_loss)
            self.train_stats['val_auc'].append(val_auc)
            
            # Print results
            logger.info(
                f"Train Loss: {train_loss:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}"
            )
            
            # Save best model
            if val_auc > best_auc:
                logger.info(f"Found better model! AUC improved from {best_auc:.4f} to {val_auc:.4f}")
                best_auc = val_auc
                if save_path:
                    self.save_model(save_path)
                    logger.info(f"Saved best model, AUC: {val_auc:.4f}")
                else:
                    logger.warning("save_path is empty, cannot save model")
            else:
                logger.info(f"Current AUC: {val_auc:.4f}, Best AUC: {best_auc:.4f} (no improvement)")
        
        logger.info(f"Training completed! Best AUC: {best_auc:.4f}")
        
        # Ensure at least the last model is saved
        if save_path and not os.path.exists(save_path):
            logger.info("Best model not found, saving current model as final model")
            self.save_model(save_path)
        
        return self.train_stats
    
    def save_model(self, path):
        """Save model"""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_stats': self.train_stats
        }, path)
    
    def load_model(self, path):
        """Load model"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.train_stats = checkpoint['train_stats']

def train_pref_model(args, train_data, valid_data, image_features, wandb_log=None, model_type='DIN'):
    """
    Main function for training preference model (supports DIN and SASRec)
    
    Args:
        args: Training parameters
        train_data: Training data
        valid_data: Validation data
        image_features: Image features
        wandb_log: wandb logger
        model_type: Model type, 'DIN' for DIN, 'SASRec' for SASRec
    Returns:
        trained_model: Trained model
        model_path: Model save path
    """
    
    if model_type == 'SASRec':
        logger.info(f"🎯 Start training SASRec preference model...")
    elif model_type == 'DIN':
        logger.info(f"🎯 Start training DIN preference model...")
    else:
        raise ValueError(f"Invalid model type: {model_type}")
    # Create dataset
    logger.info("📊 Building training dataset...")
    train_dataset = PrefDataset(args, train_data, image_features)
    val_dataset = PrefDataset(args, valid_data, image_features)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, 
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, 
        shuffle=False, num_workers=0
    )
    
    logger.info(f"Training set size: {len(train_dataset)}, Validation set size: {len(val_dataset)}")
    
    # Create model based on model_type
    if model_type == 'SASRec':

        
        model = SASRec(
            item_image_feature_dim=image_features.size(1),
            hidden_dim=args.hidden_dim,
            num_layers=args.SASRec_num_layers,
            num_heads=args.SASRec_num_heads,
            max_seq_len=args.max_seq_len,
            dropout=args.SASRec_dropout
        )
        logger.info(f"Created SASRec model: hidden_dim={args.hidden_dim}, "
                   f"num_layers={args.SASRec_num_layers}, num_heads={args.SASRec_num_heads}, max_seq_len={args.max_seq_len}")
    elif model_type == 'DIN':
        # Create DIN model
        model = DIN(
            item_image_feature_dim=image_features.size(1),
            hidden_dim=args.hidden_dim
        )
        logger.info(f"Created DIN model: hidden_dim={args.hidden_dim}")
    
    
    # Create trainer
    trainer = PreferenceTrainer(model, args, device=args.device, model_type=model_type)
    
    # Set save path
    pref_model_path = os.path.join(args.save_folder, args.pref_model_type+"_model.pt")
    
    # Training
    train_stats = trainer.train(train_loader, val_loader, save_path=pref_model_path)
    
    # Log to wandb
    if wandb_log:
        for epoch, (train_loss, val_loss, val_auc) in enumerate(
            zip(train_stats['train_loss'], train_stats['val_loss'], train_stats['val_auc'])
        ):
            wandb_log.log({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_auc": val_auc,
                "epoch": epoch
            })
    
    logger.info(f"✅ {model_type} preference model training completed! Model saved to: {pref_model_path}")
    
    return model, pref_model_path

def parse_args():
    
    parser = argparse.ArgumentParser(description="Configuration for model and training parameters")
    parser.add_argument("--device", type=str, default="cpu", help="Specify the device, e.g., 'cuda' or 'cpu'")
    
    # Dataset configuration
    parser.add_argument("--scene", type=str, default='movie',  help="dataset scene")
    parser.add_argument("--data_name", type=str)
    parser.add_argument("--data_path", type=str, default="./data/MovieLens", help="Path to the dataset")
    parser.add_argument("--save_folder", type=str, default="", help="Folder to save results")
    parser.add_argument("--item_num", type=int, default=1000, help="Number of items")

    # Feature paths
    parser.add_argument("--image_feat_path", type=str, help="Path to image features")
    parser.add_argument("--diffusion_clip_path", type=str, default="openai/clip-vit-large-patch14", help="CLIP image encoder used to extract item image features")
    parser.add_argument("--text_clip_path", type=str, default="openai/clip-vit-large-patch14", help="CLIP text encoder used to build the semantic retrieval index (paper uses ViT-H-14)")
    
    parser.add_argument("--pref_model_type", type=str, default="DIN", help="Preference model type")
    parser.add_argument("--lr",type=float, default=1e-3,help="learning rate for preference_model")
    parser.add_argument("--wd",type=float, default=1e-4,help="weight decay for preference_model")
    parser.add_argument("--epochs",type=int, default=15,help="epochs for preference_model training")
    parser.add_argument("--batch_size",type=int, default=64,help="batch size for preference_model training")
    parser.add_argument("--hidden_dim",type=int, default=128,help="hidden dimension for preference_model")
    # SASRec specific parameters
    parser.add_argument("--SASRec_num_layers", type=int, default=2, help="Number of Transformer layers for SASRec")
    parser.add_argument("--SASRec_num_heads", type=int, default=4, help="Number of attention heads for SASRec")
    parser.add_argument("--SASRec_dropout", type=float, default=0.2, help="Dropout rate for SASRec")
    parser.add_argument("--max_seq_len", type=int, default=15, help="Maximum sequence length for SASRec")

    parser.add_argument("--random_seed", type=int, default=42, help="Random seed for training")
    parser.add_argument("--use_wandb", action='store_true', help="Whether to use wandb for logging")
    


    args = parser.parse_args()
    return args
if __name__ == "__main__":
    args = parse_args()
    set_random_seeds(args.random_seed)
    args.device = torch.device("cuda") if torch.cuda.is_available() else  torch.device("cpu")
    os.makedirs(args.save_folder, exist_ok=True)
    # Initialize wandb logging
    wandb_log = None
    if args.use_wandb:
        wandb_log = wandb.init(
            project="UDPIG",
            name=f"{args.data_name}_{args.scene}_{data_utils.now_time()}",
            config={
                "scene": args.scene,   
                "pref_model_type": args.pref_model_type,
            }
        )
        print(f"✅ Wandb initialized: {wandb_log.name}")
    else:
        print("⚠️ Wandb logging disabled")
    

    

    # Prepare data and features
    print("-------------load data and feature------------")
    train_data,valid_data,test_data = load_data(args) 
    image_features = get_image_feature(args).to(args.device)
    assert image_features.shape[0]==args.item_num
    # Train preference model
    pref_model, pref_model_path = train_pref_model(
        args, train_data, valid_data, image_features, wandb_log, 
        model_type=args.pref_model_type
    )
    # Get top k_r semantically similar images for each image
    get_top_k_similar_images(args,args.data_path,k_r=100)

