#!/usr/bin/env python3
"""
Training script for preference models 
Train the preference model before preference reinforcement learning

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
from sklearn.metrics import accuracy_score,  roc_auc_score
import torch.nn.functional as F
from data_utils import *
def setup_logger():
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    if not logger.handlers:
        logger.addHandler(console_handler)

setup_logger()
logger = logging.getLogger(__name__)

class AttentionLayer(nn.Module):
    def __init__(self, hidden_dim):
        super(AttentionLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.attention = nn.Sequential(
            nn.Linear(4 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
    def forward(self, user_behavior, target_item, mask=None):
        target_item = target_item.unsqueeze(1).repeat(1, user_behavior.size(1), 1)
        interaction = torch.cat([
            user_behavior,
            target_item,
            user_behavior - target_item,
            user_behavior * target_item
        ], dim=-1)
        att_weights = self.attention(interaction)
        
        if mask is not None:
            mask = mask.unsqueeze(-1)
            att_weights = att_weights.masked_fill(mask == 0, -1e9)
            
        att_weights = F.softmax(att_weights, dim=1)
        user_interest = torch.sum(att_weights * user_behavior, dim=1)
        return user_interest, att_weights
class DIN(nn.Module):
    def __init__(self, item_image_feature_dim, hidden_dim=128):
        super(DIN, self).__init__()
        self.hidden_dim = hidden_dim
        
        # Image feature processing layer - use image features only
        self.image_feature_processor = nn.Sequential(
            nn.Linear(item_image_feature_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Attention layer
        self.attention = AttentionLayer(hidden_dim)
        
        # Final prediction layer: use only user interest features and target item features
        self.predict_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),  # Interest features + target item features
            nn.LayerNorm(hidden_dim),  
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, item_seq_images, target_item_image, mask=None):
        """
        Forward propagation using only image features
        """
        # Process target item image features / generated image features
        target_image_emb = self.image_feature_processor(target_item_image)  # shape=(batch_size, hidden_dim)
        
        # Process user behavior sequence item image features
        batch_size, seq_len = item_seq_images.size(0), item_seq_images.size(1)
        
        # Image feature processing
        item_seq_images_flat = item_seq_images.view(batch_size * seq_len, -1)  # shape=(batch*seq, image_dim)
        item_seq_image_emb_flat = self.image_feature_processor(item_seq_images_flat)  # shape=(batch*seq, hidden_dim)
        item_seq_emb = item_seq_image_emb_flat.view(batch_size, seq_len, -1)  # shape=(batch, seq, hidden_dim)
        
        # Apply attention mechanism
        user_interest, att_weights = self.attention(item_seq_emb, target_image_emb, mask)
        
        # Concatenate user interest features and target item features
        concat_features = torch.cat([user_interest, target_image_emb], dim=-1)
        
        # Predict CTR logits
        output = self.predict_layer(concat_features)  # shape=(batch_size, 1)
        
        # In eval mode, return probability for inference
        if not self.training:
            output = torch.sigmoid(output)
        
        return output, att_weights

# ===================== SASRec Model =====================
class SASRec(nn.Module):
    """
    SASRec-based CTR prediction model
    Use Transformer to encode user history sequence, combine with target item to predict click probability
    """
    def __init__(self, item_image_feature_dim, hidden_dim=128, num_layers=2, num_heads=4, 
                 max_seq_len=15, dropout=0.2):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.max_seq_len = max_seq_len
        
        # 1. Image feature mapping layer
        self.image_feature_processor = nn.Sequential(
            nn.Linear(item_image_feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # 2. Position embedding
        self.position_embedding = nn.Embedding(max_seq_len, hidden_dim)
        
        # 3. Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # 4. Attention pooling
        self.attention_pooling = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1, bias=False)
        )
        
        # 5. Prediction layer
        self.predict_layer = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # Use smaller initialization std to avoid gradient explosion
            nn.init.xavier_uniform_(module.weight, gain=0.5)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0.0)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.01)
        elif isinstance(module, nn.LayerNorm):
            nn.init.constant_(module.bias, 0.0)
            nn.init.constant_(module.weight, 1.0)
    
    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1).bool()
        return mask
    
    def forward(self, item_seq_images, target_item_image, mask=None):
        """
        Forward propagation
        Args:
            item_seq_images: Historical sequence image features, shape=[batch, seq_len, feat_dim]
            target_item_image: Target item image features, shape=[batch, feat_dim]
            mask: padding mask, shape=[batch, seq_len], 1 means valid, 0 means padding
        Returns:
            output: Click probability logits, shape=[batch, 1]
            att_weights: Attention weights (for compatibility with DIN interface)
        """
        batch_size, seq_len = item_seq_images.shape[0], item_seq_images.shape[1]
        device = item_seq_images.device

        # 1. Image feature mapping
        item_emb = self.image_feature_processor(item_seq_images)  # [batch, seq_len, hidden_dim]
        
        # 2. Position encoding
        position_ids = torch.arange(seq_len, device=device).unsqueeze(0).repeat(batch_size, 1)
        pos_emb = self.position_embedding(position_ids)
        x = item_emb + pos_emb
        
        # 3. Transformer encoding
        causal_mask = self.generate_causal_mask(seq_len, device)
        src_key_padding_mask = (mask == 0) if mask is not None else None
        sequence_encoding = self.transformer_encoder(
            src=x,
            mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        # 4. Attention pooling
        att_weights = self.attention_pooling(sequence_encoding)  # [batch, seq_len, 1]
        if mask is not None:
            att_weights = att_weights.masked_fill(mask.unsqueeze(-1) == 0, -1e9)
        att_weights = F.softmax(att_weights, dim=1)
        user_representation = torch.sum(att_weights * sequence_encoding, dim=1)  # [batch, hidden_dim]
        
        # 5. Process target item features
        target_emb = self.image_feature_processor(target_item_image)  # [batch, hidden_dim]
        
        # 6. Concatenate and predict
        concat_features = torch.cat([user_representation, target_emb], dim=-1)
        output = self.predict_layer(concat_features)
        
        # In eval mode, return probability for inference
        if not self.training:
            output = torch.sigmoid(output)
        
        return output, att_weights

