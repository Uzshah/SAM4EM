import logging

import torch
from .modeling.backbones.image_encoder import ImageEncoder, FpnNeck
from .modeling.backbones.hieradet import Hiera
from .modeling.position_encoding import PositionEmbeddingSine
from .modeling.sam2 import Sam
from .modeling.sam import MaskDecoder, PromptEncoder, TwoWayTransformer
from .modeling.sam.image_encoder import ImageEncoderViT
from functools import partial
from torch.nn import functional as F
from .modeling.sam.transformer import RoPEAttention
from .modeling import memory_attention, memory_encoder
from .modeling.position_encoding import PositionEmbeddingSine


def build_sam(
        device="cuda",
        mode="eval",
        img_size = 1024, num_classes = 2,
        checkpoint_sam1 = None,
        is_prompt_training = False,
        is_double_masking = False,
        adapter: bool = False,
        embedd = False,
        checkpoint_sam2 = None,
        mem_slot = 4,
        mem_avg_parm = 0.1,
    ):
    # if img_encoder=="sam2":
    trunk = Hiera(embed_dim=112, num_heads= 2)
    position_encoding = PositionEmbeddingSine(num_pos_feats= 256, normalize = True, temperature= 1000)
    neck = FpnNeck(position_encoding = position_encoding, d_model = 256, backbone_channel_list= [896, 448, 224, 112],
                  fpn_top_down_levels= [2, 3],  # output level 0 and 1 directly use the backbone features
                  fpn_interp_model= 'nearest')
    sam_encoder = ImageEncoder(trunk = trunk, neck = neck, img_size = img_size, scalp =1)
    
    sam = Sam(
        image_encoder = sam_encoder,
        prompt_encoder1=PromptEncoder(
            embed_dim=256,
            image_embedding_size=(img_size//4, img_size//4),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            ),
        mask_decoder1=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
              
            ),
        prompt_encoder2 = PromptEncoder(
            embed_dim=256,
            image_embedding_size=(img_size//4, img_size//4),
            input_image_size=(img_size, img_size),
            mask_in_chans=16,
            ) if is_double_masking else None,
        mask_decoder2=MaskDecoder(
            num_multimask_outputs=num_classes,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=256,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=256,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
            is_second = False,
            )if is_double_masking else None,

        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
        mem_slot = mem_slot,
        mem_avg_parm =mem_avg_parm,
    )
    sam.eval()
    if checkpoint_sam2 is not None:
        with open(checkpoint_sam2, "rb") as f:
            state_dict = torch.load(f)['model']
        try:
            sam.load_state_dict(state_dict)
        except:
            new_state_dict = load_from_sam2(sam, state_dict)
            sam.load_state_dict(new_state_dict, strict=True)
    return sam
    
        
def load_from_sam2(sam, state_dict):
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    updated_state_dict = {}
    for key, value in state_dict.items():
        if 'sam_mask_decoder.' in key:
            # Create new keys for both 'mask_decoder' and 'mask_decoder2'
            new_key2 = key.replace('sam_mask_decoder', 'mask_decoder1')
            new_key2 = key.replace('sam_mask_decoder', 'mask_decoder2')
            
            # Assign the value to both new keys
            updated_state_dict[new_key2] = value
            updated_state_dict[key] = value
    
        if 'sam_prompt_encoder.' in key:
            # Create new keys for 'prompt_encoder' and 'prompt_encoder2'
            new_key2 = key.replace('sam_prompt_encoder', 'prompt_encoder1')
            new_key2 = key.replace('sam_prompt_encoder', 'prompt_encoder2')
            
            # Assign the value to the new key and keep the original key
            updated_state_dict[key] = value
            updated_state_dict[new_key2] = value
        else:
            # For other keys, keep them as they are
            updated_state_dict[key] = value
    # print(updated_state_dict.keys())
    except_keys = ["mask_tokens", "output_hypernetworks_mlps", "iou_prediction_head"]
    # print(updated_state_dict.keys())
    # sam_dict = model.state_dict()
    new_state_dict = {k: v for k, v in updated_state_dict.items() if
                          k in sam_dict.keys() if all(exclude not in k for exclude in ["mask_tokens", 
                                                                                       "output_hypernetworks_mlps", "iou_prediction_head"])}
    sam_dict.update(new_state_dict)
    return sam_dict

def load_from_sam1(sam, state_dict, image_size, vit_patch_size, encoder_global_attn_indexes):
    ega = encoder_global_attn_indexes
    sam_dict = sam.state_dict()
    except_keys = ['mask_tokens', 'output_hypernetworks_mlps', 'iou_prediction_head']
    updated_state_dict = {}
    for key, value in state_dict.items():
        if 'mask_decoder.' in key:
            # Create new keys for both 'mask_decoder' and 'mask_decoder2'
            new_key1 = key.replace('mask_decoder', 'sam_mask_decoder')
            new_key2 = key.replace('mask_decoder', 'mask_decoder2')
            
            # Assign the value to both new keys
            updated_state_dict[new_key1] = value
            updated_state_dict[new_key2] = value
    
        if 'prompt_encoder.' in key:
            # Create new keys for 'prompt_encoder' and 'prompt_encoder2'
            new_key1 = key.replace('prompt_encoder', 'sam_prompt_encoder')
            new_key2 = key.replace('prompt_encoder', 'prompt_encoder2')
            
            # Assign the value to the new key and keep the original key
            updated_state_dict[new_key1] = value
            updated_state_dict[new_key2] = value
        else:
            # For other keys, keep them as they are
            updated_state_dict[key] = value
    new_state_dict = {k: v for k, v in updated_state_dict.items() if
                      k in sam_dict.keys() and except_keys[0] not in k and except_keys[1] not in k and except_keys[2] not in k}
    pos_embed = new_state_dict['image_encoder.pos_embed']
    token_size = int(image_size // vit_patch_size)
    if pos_embed.shape[1] != token_size:
        # resize pos embedding, which may sacrifice the performance, but I have no better idea
        pos_embed = pos_embed.permute(0, 3, 1, 2)  # [b, c, h, w]
        pos_embed = F.interpolate(pos_embed, (token_size, token_size), mode='bilinear', align_corners=False)
        pos_embed = pos_embed.permute(0, 2, 3, 1)  # [b, h, w, c]
        new_state_dict['image_encoder.pos_embed'] = pos_embed
        rel_pos_keys = [k for k in sam_dict.keys() if 'rel_pos' in k]

        global_rel_pos_keys = []
        for rel_pos_key in rel_pos_keys:
            num = int(rel_pos_key.split('.')[2])
            if num in encoder_global_attn_indexes:
                global_rel_pos_keys.append(rel_pos_key)
        # global_rel_pos_keys = [k for k in rel_pos_keys if '2' in k or '5' in  k or '8' in k or '11' in k]
        for k in global_rel_pos_keys:
            rel_pos_params = new_state_dict[k]
            h, w = rel_pos_params.shape
            rel_pos_params = rel_pos_params.unsqueeze(0).unsqueeze(0)
            rel_pos_params = F.interpolate(rel_pos_params, (token_size * 2 - 1, w), mode='bilinear', align_corners=False)
            new_state_dict[k] = rel_pos_params[0, 0, ...]
    sam_dict.update(new_state_dict)
    return sam_dict
    
        