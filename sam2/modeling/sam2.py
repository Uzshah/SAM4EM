import torch
from torch import nn
from torch.nn import functional as F

from typing import Any, Dict, List, Tuple

from .sam2_utils import LayerNorm2d
from ..utils.transforms import SAM2Transforms

from scipy.ndimage import label
from skimage.measure import regionprops
import numpy as np
import cv2
import math



class DoubleAttentionLayer(nn.Module):
    def __init__(self, embed_dim=256, num_heads=8, max_len=2048):
        super(DoubleAttentionLayer, self).__init__()
        self.attention1 = nn.MultiheadAttention(embed_dim, num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim, num_heads)
        
        # LayerNorm and Feed-forward layer for added stability
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.feedforward = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.ReLU(),
            nn.Linear(embed_dim * 2, embed_dim),
        )
        
        # Sine-based Positional Encoding
        self.register_buffer('positional_encoding', self._get_sine_positional_encoding(embed_dim, max_len))
    
    def _get_sine_positional_encoding(self, embed_dim, max_len):
        """
        Generate sine-based positional encoding.
        """
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2, dtype=torch.float32) * (-math.log(10000.0) / embed_dim))
        pe = torch.zeros(max_len, 1, embed_dim)
        pe[:, 0, 0::2] = torch.sin(pos * div_term)
        pe[:, 0, 1::2] = torch.cos(pos * div_term)
        return pe
        
    def forward(self, x):
        # Reshape for attention
        batch_size, channels, h, w = x.size()
        x = x.view(batch_size, channels, -1).permute(2, 0, 1)  # Shape: (32*32, batch, 256)
        
        # Add positional encoding
        x = x + self.positional_encoding[:x.size(0), :, :].to(x.device)
        
        # First attention layer
        attn_output, _ = self.attention1(x, x, x)
        x = self.norm1(attn_output)
        
        # Second attention layer
        attn_output, _ = self.attention2(x, x, x)
        x = self.norm2(attn_output)
        
        # Feedforward layer
        x = x + self.feedforward(x)
        
        # Reshape back to original
        x = x.permute(1, 2, 0).view(batch_size, channels, h, w)
        
        return x



class ConvSequence(nn.Module):
    def __init__(self, embed_dim=256):
        super(ConvSequence, self).__init__()
        
        self.conv_sequence = nn.Sequential(
            # First conv block
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                stride = 2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            
            # Second conv block
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            
            # Third conv block
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                stride = 2,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
            
            # # Fourth conv block
            nn.Conv2d(
                in_channels=embed_dim,
                out_channels=embed_dim,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(embed_dim),
            nn.GELU(),
        )
        
        # Optional: Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize the weights using Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # Input shape: (batch_size, embed_dim, h, w)
        return self.conv_sequence(x)


# Alternative version with residual connections
class FPNFusionResidual(nn.Module):
    def __init__(self, in_channels=256):
        super(FPNFusionResidual, self).__init__()
        
        # Smoothing convs with residual connections
        self.smooth_32 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        # self.parser32 = FFParser(256, 32, 32)
        self.smooth_64 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
        )
        # self.parser64 = FFParser(256, 64, 64)
        # Final fusion with residual connection
        self.fusion = nn.Sequential(
            # Initial channel reduction from in_channels*3 to in_channels
            nn.Conv2d(in_channels * 3, in_channels, kernel_size=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # First transpose conv: upscale by 2
            nn.Conv2d(in_channels, in_channels, 
                              kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            
            # Second transpose conv: upscale by 2 again (total 4x)
            nn.Conv2d(in_channels, in_channels, 
                              kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.downsampling_128 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        self.downsampling_64 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels)
        )
        # self.parser128 = FFParser(256, 128, 128)#
        self.relu = nn.ReLU(inplace=True)
        # self.norm = nn.BatchNorm2d(in_channels)
        self.out_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1)
        )
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, p3, p4, p5):
        """
        Args:
            p3: Feature map (B, 256, 128, 128)
            p4: Feature map (B, 256, 64, 64)
            p5: Feature map (B, 256, 32, 32)
        Returns:
            fused_features: Combined feature map (B, 256, 128, 128)
        """
        up_32 = F.interpolate(p5, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        up_32 = self.relu(up_32 + self.smooth_32(up_32))
        up_64 = F.interpolate(p4, size=p3.shape[-2:], mode='bilinear', align_corners=False)
        up_64 = self.relu(up_64 + self.smooth_64(up_64))

        concat_features = torch.cat([
            p3,
            up_64,
            up_32,
        ], dim=1)
        
        # Final fusion with residual connection
        out = self.fusion(concat_features)
        out = self.relu(out + p3)  # Residual connection with highest resolution features
        out = self.out_proj(out)
        return out



class Sam(nn.Module):
    mask_threshold: float = 0.0
    image_format: str = "RGB"

    def __init__(
            self,
            image_encoder,
            prompt_encoder1 = None,
            mask_decoder1 = None,
            prompt_encoder2 = None,
            mask_decoder2 = None,
            memory_attention = None,
            memory_encoder = None,
            pixel_mean: List[float] = [123.675, 116.28, 103.53],
            pixel_std: List[float] = [58.395, 57.12, 57.375],
            mem_slot = 8,
            mem_avg_parm = 0.3,
        ) -> None:

        super().__init__()
        
        self.image_encoder = image_encoder
        self.prompt_encoder = prompt_encoder1
        self.mask_decoder = mask_decoder1
        self.prompt_encoder2 = prompt_encoder2
        self.mask_decoder2 = mask_decoder2
        self.mem_slot = mem_slot
        self.mem_avg_parm = mem_avg_parm
        # self.learnable_param1 = nn.Parameter(torch.rand(1, requires_grad=True))
        # self.learnable_param2 = nn.Parameter(torch.rand(1, requires_grad=True))
        self.fpn = FPNFusionResidual()
        self.mem = create_memory_encoder(mem_slot = mem_slot, mem_avg_parm = mem_avg_parm)
            
    def get_threshold(self):
        return torch.sigmoid(self.learnable_param1), torch.sigmoid(self.learnable_param2)
        
    def forward(
        self,
        batched_input: List[Dict[str, Any]],
        multimask_output: bool=True,
        mode = "train",
        previous_mask= None,
        memory_encoded = None,
    ) -> List[Dict[str, torch.Tensor]]:
        
        backbone_out = self.image_encoder(batched_input)
        image_embeddings = backbone_out['vision_features']
        # print(len( backbone_out['backbone_fpn']))
        x1, x2, x3 = backbone_out['backbone_fpn']
        x1_pos, x2_pos, x3_pos = backbone_out['vision_pos_enc']
        image_embeddings = self.fpn(x1+x1_pos, x2 +x2_pos, x3 + x3_pos)

        outputs = []
        outputs2 = []
        dense_output = []
        mem_feat_list = []
        mem_pos_list = []
        low_res_masks2 = None
        low_res_masks = None
        output_corse =None
        
        for idx, curr_embedding in enumerate(image_embeddings):
            
            if low_res_masks2 is not None:
                # constrained_param1 = torch.sigmoid(self.learnable_param1)
                masks = (torch.sigmoid(low_res_masks2.clone().detach())>0.5).float()
            else:
                masks = previous_mask[0].unsqueeze(0).unsqueeze(0).float() 
                    
                max_flage = False
                points = None
                boxes = None
            
            curr_embedding = self.mem(curr_embedding.unsqueeze(0), masks)
            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                        points=points,
                        boxes=boxes,
                        masks= masks,
                    )
            # print(curr_embedding.shape)
            low_res_masks, iou_predictions,msk_feat, up_embedd = self.mask_decoder(
                    image_embeddings=curr_embedding,
                    image_pe=self.prompt_encoder.get_dense_pe(),
                    sparse_prompt_embeddings=sparse_embeddings,
                    dense_prompt_embeddings=dense_embeddings,
                    multimask_output=False,
                    repeat_image = True,
                )
            if max_flage:
                low_res_masks = self._log_sum_exp(low_res_masks)
                
            if self.mask_decoder2 is not None:
                # constrained_param2 = torch.sigmoid(self.learnable_param2)
                mask2 = (torch.sigmoid(low_res_masks)>0.5).float()    
                sparse_embeddings2, dense_embeddings2 = self.prompt_encoder2(
                        points=points,
                        boxes=boxes,
                        masks= mask2,
                    )
                low_res_masks2, iou_predictions,mask_feat,_ = self.mask_decoder2(
                        image_embeddings=curr_embedding,
                        image_pe=self.prompt_encoder.get_dense_pe(),
                        sparse_prompt_embeddings=sparse_embeddings,
                        dense_prompt_embeddings=dense_embeddings,
                        multimask_output=False,
                        repeat_image = True,
                        up_embedd = up_embedd,
                    )
                outputs2.append(low_res_masks2.squeeze(1))
            outputs.append(low_res_masks.squeeze(1))
            

        return torch.stack(outputs, dim=0), torch.stack(outputs2, dim=0) if self.mask_decoder2 is not None else None
        
    
        
    def _log_sum_exp(self, x):
        max_x = torch.max(x, dim=0, keepdim=True)[0]
        return max_x + torch.log(torch.sum(torch.exp(x - max_x), dim=0, keepdim=True))
        
    def postprocess_masks(
        self,
        masks: torch.Tensor,
        input_size: Tuple[int, ...],
        original_size: Tuple[int, ...],
    ) -> torch.Tensor:
        """
        Remove padding and upscale masks to the original image size.

        Arguments:
          masks (torch.Tensor): Batched masks from the mask_decoder,
            in BxCxHxW format.
          input_size (tuple(int, int)): The size of the image input to the
            model, in (H, W) format. Used to remove padding.
          original_size (tuple(int, int)): The original size of the image
            before resizing for input to the model, in (H, W) format.

        Returns:
          (torch.Tensor): Batched masks in BxCxHxW format, where (H, W)
            is given by original_size.
        """
        masks = F.interpolate(
            masks,
            (self.image_encoder.img_size, self.image_encoder.img_size),
            mode="bilinear",
            align_corners=False,
        )
        masks = masks[..., : input_size[0], : input_size[1]]
        masks = F.interpolate(masks, original_size, mode="bilinear", align_corners=False)
        return masks



def create_positional_encoding(d_model, height, width):
    """Create 2D positional encoding (remains unchanged)"""
    if d_model % 4 != 0:
        raise ValueError("d_model must be divisible by 4 for 2D positional encoding")
        
    pe = torch.zeros(1, d_model, height, width)
    d_model = int(d_model / 2)
    
    y_pos = torch.arange(height).float().unsqueeze(1).expand(-1, width)
    x_pos = torch.arange(width).float().unsqueeze(0).expand(height, -1)
    
    div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
    
    pe[0, 0:d_model:2, :, :] = torch.sin(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
    pe[0, 1:d_model:2, :, :] = torch.cos(y_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
    
    pe[0, d_model::2, :, :] = torch.sin(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
    pe[0, d_model+1::2, :, :] = torch.cos(x_pos.unsqueeze(0) * div_term.view(-1, 1, 1))
    
    return pe

class LightweightMemoryEncoder(nn.Module):
    def __init__(self, feat_channels=256, feat_size=(128, 128), 
                 memory_dim=128, num_memory_slots=8, mem_avg_parm=0.1,
                 dropout_rate=0.1, feature_dropout=0.1, attention_dropout=0.1):
        """
        Lightweight memory encoder with positional encoding and dropout
        
        Args:
            feat_channels: Number of feature channels (256)
            feat_size: Spatial size of feature maps (128, 128)
            memory_dim: Dimension of memory slots (reduced for efficiency)
            num_memory_slots: Number of memory slots (reduced for efficiency)
            mem_avg_parm: Memory averaging parameter
            dropout_rate: Dropout rate for general features
            feature_dropout: Dropout rate for feature projection
            attention_dropout: Dropout rate for attention
        """
        super().__init__()
        
        self.feat_channels = feat_channels
        self.feat_size = feat_size
        self.memory_dim = memory_dim
        self.mem_avg_parm = mem_avg_parm
        self.num_memory_slots = num_memory_slots
        
        # Create and register positional encoding
        pe = create_positional_encoding(memory_dim, feat_size[0], feat_size[1])
        self.register_buffer('positional_encoding', pe)
        
        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)
        self.feature_dropout = nn.Dropout2d(feature_dropout)  # Spatial dropout
        self.attention_dropout = nn.Dropout(attention_dropout)
        
        # Lightweight mask encoder with dropout
        self.mask_encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate),
            nn.Conv2d(32, memory_dim, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
        
        # Feature projection with dropout
        self.feature_proj = nn.Sequential(
            nn.Conv2d(feat_channels, memory_dim, 1),
            nn.ReLU(),
            nn.Dropout2d(feature_dropout)
        )
        
        # Single memory
        self.register_buffer('memory',
                           torch.zeros(num_memory_slots, memory_dim, feat_size[0], feat_size[1]))
        
        # Attention layers with dropout
        self.key_proj = nn.Sequential(
            nn.Conv2d(memory_dim, memory_dim, 1),
            nn.Dropout2d(feature_dropout)
        )
        self.query_proj = nn.Sequential(
            nn.Conv2d(memory_dim, memory_dim, 1),
            nn.Dropout2d(feature_dropout)
        )
        self.value_proj = nn.Sequential(
            nn.Conv2d(memory_dim, memory_dim, 1),
            nn.Dropout2d(feature_dropout)
        )
        
        # Output projection with dropout
        self.output_proj = nn.Sequential(
            nn.Conv2d(memory_dim, feat_channels, 1),
            nn.ReLU(),
            nn.Dropout2d(dropout_rate)
        )
    
    def _add_positional_encoding(self, x):
        """Add positional encoding to input tensor"""
        return x + self.positional_encoding
    
    def _efficient_attention(self, query, key, value):
        """
        Compute efficient spatial attention with dropout
        """
        # Add positional encoding
        query = self._add_positional_encoding(query)
        key = self._add_positional_encoding(key)
        
        B, C, H, W = query.shape
        M = key.shape[0]
        
        # Apply dropout to query and key
        query = self.attention_dropout(query)
        key = self.attention_dropout(key)
        
        # Compute attention scores
        attn = torch.einsum('bchw,mchw->bmhw', query.view(B, C, H, W), 
                           key.view(M, C, H, W)) / torch.sqrt(torch.tensor(C, dtype=torch.float32))
        attn = F.softmax(attn, dim=1)
        
        # Apply attention dropout
        attn = self.attention_dropout(attn)
        
        # Apply attention to values
        out = torch.einsum('bmhw,mchw->bchw', attn, value.view(M, C, H, W))
        
        return out
    
    def update_memory(self, feat, mask):
        """
        Efficient memory update with dropout
        """
        # Project inputs with dropout
        proj_feat = self.feature_proj(feat)
        mask_feat = self.mask_encoder(mask)
        
        # Combine feature and mask information
        combined_feat = proj_feat + F.interpolate(mask_feat, size=proj_feat.shape[2:])
        combined_feat = self.feature_dropout(combined_feat)
        
        # Add positional encoding
        combined_feat = self._add_positional_encoding(combined_feat)
        
        # Generate attention components
        queries = self.query_proj(combined_feat)
        keys = self.key_proj(self.memory)
        values = self.value_proj(self.memory)
        
        # Compute attention and update memory
        attn_output = self._efficient_attention(queries, keys, values)
        
        # Update memory with moving average
        with torch.no_grad():
            update = combined_feat.mean(dim=0, keepdim=True)
            self.memory = self.memory * (1-self.mem_avg_parm) + update * self.mem_avg_parm
        
        return attn_output
    
    def forward(self, features, prev_masks):
        """Forward pass with dropout"""
        # Apply initial feature dropout
        features = self.feature_dropout(features)
        
        # Get memory-enhanced features
        memory_output = self.update_memory(features, prev_masks)
        
        # Enhance original features with memory information
        enhanced_features = features + self.output_proj(memory_output)
        
        return enhanced_features

def create_memory_encoder(mem_slot=8, mem_avg_parm=0.1, dropout_rate=0.1):
    """Create lightweight memory encoder with dropout"""
    model = LightweightMemoryEncoder(
        feat_channels=256,
        feat_size=(128, 128),
        memory_dim=128,
        num_memory_slots=mem_slot,
        mem_avg_parm=mem_avg_parm,
        dropout_rate=dropout_rate,
        feature_dropout=dropout_rate,
        attention_dropout=dropout_rate
    )
    return model

# Example usage
if __name__ == "__main__":
    # Create sample data
    batch_size = 2
    features = torch.randn(batch_size, 256, 128, 128)
    prev_masks = torch.randint(0, 2, (batch_size, 1, 512, 512), dtype=torch.float32)
    
    # Initialize model
    model = create_memory_encoder()
    
    # Get enhanced features
    enhanced_features = model(features, prev_masks)
    print(f"Enhanced feature shape: {enhanced_features.shape}")
    
    # Calculate model size
    param_size = sum(p.numel() for p in model.parameters()) * 4 / 1024 / 1024  # Size in MB
    print(f"Model size: {param_size:.2f} MB")
