# from .segment_anything import build_sam, SamPredictor
# from .segment_anything import sam_model_registry

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
# from .segment_anything.modeling import Sam
from safetensors import safe_open
from safetensors.torch import save_file

from icecream import ic


class _LoRA_qkv(nn.Module):
    """In Sam it is implemented as
    self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
    B, N, C = x.shape
    qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    q, k, v = qkv.unbind(0)
    """

    def __init__(
            self,
            qkv: nn.Module,
            linear_a_q: nn.Module,
            linear_b_q: nn.Module,
            linear_a_v: nn.Module,
            linear_b_v: nn.Module,
    ):
        super().__init__()
        self.qkv = qkv
        self.linear_a_q = linear_a_q
        self.linear_b_q = linear_b_q
        self.linear_a_v = linear_a_v
        self.linear_b_v = linear_b_v
        self.dim = qkv.in_features
        self.w_identity = torch.eye(qkv.in_features)

    def forward(self, x):
        qkv = self.qkv(x)  # B,N,N,3*org_C
        new_q = self.linear_b_q(self.linear_a_q(x))
        new_v = self.linear_b_v(self.linear_a_v(x))
        qkv[:, :, :, : self.dim] += new_q
        qkv[:, :, :, -self.dim:] += new_v
        return qkv


class LoRA_Sam(nn.Module):
    """Applies low-rank adaptation to a Sam model's image encoder.

    Args:
        sam_model: a vision transformer model, see base_vit.py
        r: rank of LoRA
        num_classes: how many classes the model output, default to the vit model
        lora_layer: which layer we apply LoRA.

    Examples::
        >>> model = ViT('B_16_imagenet1k')
        >>> lora_model = LoRA_ViT(model, r=4)
        >>> preds = lora_model(img)
        >>> print(preds.shape)
        torch.Size([1, 1000])
    """

    def __init__(self, sam_model, r: int, lora_layer=None):
        super(LoRA_Sam, self).__init__()

        assert r > 0
        # base_vit_dim = sam_model.image_encoder.patch_embed.proj.out_channels
        # dim = base_vit_dim
        if lora_layer:
            self.lora_layer = lora_layer
        else:
            self.lora_layer = list(
                range(len(sam_model.image_encoder.blocks)))  # Only apply lora to the image encoder by default
        # create for storage, then we can init them or load weights
        self.w_As = []  # These are linear layers
        self.w_Bs = []

        # lets freeze first
        for param in sam_model.image_encoder.parameters():
            param.requires_grad = False

        # Here, we do the surgery
        for t_layer_i, blk in enumerate(sam_model.image_encoder.blocks):
            # If we only want few lora layer instead of all
            if t_layer_i not in self.lora_layer:
                continue
            w_qkv_linear = blk.attn.qkv
            self.dim = w_qkv_linear.in_features
            w_a_linear_q = nn.Linear(self.dim, r, bias=False)
            w_b_linear_q = nn.Linear(r, self.dim, bias=False)
            w_a_linear_v = nn.Linear(self.dim, r, bias=False)
            w_b_linear_v = nn.Linear(r, self.dim, bias=False)
            self.w_As.append(w_a_linear_q)
            self.w_Bs.append(w_b_linear_q)
            self.w_As.append(w_a_linear_v)
            self.w_Bs.append(w_b_linear_v)
            blk.attn.qkv = _LoRA_qkv(
                w_qkv_linear,
                w_a_linear_q,
                w_b_linear_q,
                w_a_linear_v,
                w_b_linear_v,
            )

        # for param in sam_model.mask_decoder2.transformer.parameters():
        #     param.requires_grad = False
        # decoder_transformer = sam_model.mask_decoder2.transformer
        # for layer_idx, blk in enumerate(decoder_transformer.layers):
        #     self_attn_q_proj = blk.self_attn.q_proj
        #     self_attn_v_proj = blk.self_attn.v_proj
        #     input_dim = blk.self_attn.embedding_dim
        #     output_dim = blk.self_attn.internal_dim
        #     w_a_linear_q_self_attn = nn.Linear(input_dim, r, bias=False)
        #     w_b_linear_q_self_attn = nn.Linear(r, output_dim, bias=False)
        #     w_a_linear_v_self_attn = nn.Linear(input_dim, r, bias=False)
        #     w_b_linear_v_self_attn = nn.Linear(r, output_dim, bias=False)
        #     self.self_attn_As.append(w_a_linear_q_self_attn)
        #     self.self_attn_Bs.append(w_b_linear_q_self_attn)
        #     self.self_attn_As.append(w_a_linear_v_self_attn)
        #     self.self_attn_Bs.append(w_b_linear_v_self_attn)
        #     blk.self_attn.q_proj = _LoRA_qkv_proj(self_attn_q_proj, w_a_linear_q_self_attn, w_b_linear_q_self_attn)
        #     blk.self_attn.v_proj = _LoRA_qkv_proj(self_attn_v_proj, w_a_linear_v_self_attn, w_b_linear_v_self_attn)

        #     cross_attn_ti_q_proj = blk.cross_attn_token_to_image.q_proj
        #     cross_attn_ti_v_proj = blk.cross_attn_token_to_image.v_proj
        #     ti_input_dim = blk.cross_attn_token_to_image.embedding_dim
        #     ti_output_dim = blk.cross_attn_token_to_image.internal_dim
        #     w_a_linear_q_cross_attn_ti = nn.Linear(ti_input_dim, r, bias=False)
        #     w_b_linear_q_cross_attn_ti = nn.Linear(r, ti_output_dim, bias=False)
        #     w_a_linear_v_cross_attn_ti = nn.Linear(ti_input_dim, r, bias=False)
        #     w_b_linear_v_cross_attn_ti = nn.Linear(r, ti_output_dim, bias=False)
        #     self.cross_attn_ti_As.append(w_a_linear_q_cross_attn_ti)
        #     self.cross_attn_ti_Bs.append(w_b_linear_q_cross_attn_ti)
        #     self.cross_attn_ti_As.append(w_a_linear_v_cross_attn_ti)
        #     self.cross_attn_ti_Bs.append(w_b_linear_v_cross_attn_ti)
        #     blk.cross_attn_token_to_image.q_proj = _LoRA_qkv_proj(cross_attn_ti_q_proj, w_a_linear_q_cross_attn_ti,
        #                                                           w_b_linear_q_cross_attn_ti)
        #     blk.cross_attn_token_to_image.v_proj = _LoRA_qkv_proj(cross_attn_ti_v_proj, w_a_linear_v_cross_attn_ti,
        #                                                           w_b_linear_v_cross_attn_ti)

        #     cross_attn_it_q_proj = blk.cross_attn_image_to_token.q_proj
        #     cross_attn_it_v_proj = blk.cross_attn_image_to_token.v_proj
        #     it_input_dim = blk.cross_attn_image_to_token.embedding_dim
        #     it_output_dim = blk.cross_attn_image_to_token.internal_dim
        #     w_a_linear_q_cross_attn_it = nn.Linear(it_input_dim, r, bias=False)
        #     w_b_linear_q_cross_attn_it = nn.Linear(r, it_output_dim, bias=False)
        #     w_a_linear_v_cross_attn_it = nn.Linear(it_input_dim, r, bias=False)
        #     w_b_linear_v_cross_attn_it = nn.Linear(r, it_output_dim, bias=False)
        #     self.cross_attn_it_As.append(w_a_linear_q_cross_attn_it)
        #     self.cross_attn_it_Bs.append(w_b_linear_q_cross_attn_it)
        #     self.cross_attn_it_As.append(w_a_linear_v_cross_attn_it)
        #     self.cross_attn_it_Bs.append(w_b_linear_v_cross_attn_it)
        #     blk.cross_attn_image_to_token.q_proj = _LoRA_qkv_proj(cross_attn_it_q_proj, w_a_linear_q_cross_attn_it,
        #                                                           w_b_linear_q_cross_attn_it)
        #     blk.cross_attn_image_to_token.v_proj = _LoRA_qkv_proj(cross_attn_it_v_proj, w_a_linear_v_cross_attn_it,
        #                                                           w_b_linear_v_cross_attn_it)

        # # final attention token to image
        # block = decoder_transformer.final_attn_token_to_image
        # fa_ti_q_proj = block.q_proj
        # fa_ti_v_proj = block.v_proj
        # in_dim, out_dim = block.embedding_dim, block.internal_dim
        # self.fa_ti_q_proj_A = nn.Linear(in_dim, r, bias=False)
        # self.fa_ti_q_proj_B = nn.Linear(r, out_dim, bias=False)
        # self.fa_ti_v_proj_A = nn.Linear(in_dim, r, bias=False)
        # self.fa_ti_v_proj_B = nn.Linear(r, out_dim, bias=False)
        # block.q_proj = _LoRA_qkv_proj(fa_ti_q_proj, self.fa_ti_q_proj_A, self.fa_ti_q_proj_B)
        # block.v_proj = _LoRA_qkv_proj(fa_ti_v_proj, self.fa_ti_v_proj_A, self.fa_ti_v_proj_B)
        self.reset_parameters()
        self.sam = sam_model

    def save_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.

        save both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        num_layer = len(self.w_As)  # actually, it is half
        a_tensors = {f"w_a_{i:03d}": self.w_As[i].weight for i in range(num_layer)}
        b_tensors = {f"w_b_{i:03d}": self.w_Bs[i].weight for i in range(num_layer)}
        prompt_encoder_tensors = {}
        prompt_encoder2_tensors = {}
        mask_decoder_tensors = {}
        mask_decoder2_tensors = {}

        # save prompt encoder, only `state_dict`, the `named_parameter` is not permitted
        if isinstance(self.sam, torch.nn.DataParallel) or isinstance(self.sam, torch.nn.parallel.DistributedDataParallel):
            state_dict = self.sam.module.state_dict()
        else:
            state_dict = self.sam.state_dict()
        for key, value in state_dict.items():
            if 'prompt_encoder' in key:
                prompt_encoder_tensors[key] = value
            if 'prompt_encoder2' in key:
                prompt_encoder2_tensors[key] = value
            if 'mask_decoder' in key:
                mask_decoder_tensors[key] = value
            if 'mask_decoder2' in key:
                mask_decoder2_tensors[key] = value

        merged_dict = {**a_tensors, **b_tensors, **prompt_encoder_tensors,**prompt_encoder2_tensors, **mask_decoder_tensors, **mask_decoder2_tensors}
        torch.save(merged_dict, filename)

    def load_lora_parameters(self, filename: str) -> None:
        r"""Only safetensors is supported now.

        pip install safetensor if you do not have one installed yet.\

        load both lora and fc parameters.
        """

        assert filename.endswith(".pt") or filename.endswith('.pth')

        state_dict = torch.load(filename)

        for i, w_A_linear in enumerate(self.w_As):
            saved_key = f"w_a_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_A_linear.weight = Parameter(saved_tensor)

        for i, w_B_linear in enumerate(self.w_Bs):
            saved_key = f"w_b_{i:03d}"
            saved_tensor = state_dict[saved_key]
            w_B_linear.weight = Parameter(saved_tensor)

        sam_dict = self.sam.state_dict()
        sam_keys = sam_dict.keys()

        # load prompt encoder
        prompt_encoder_keys = [k for k in sam_keys if 'prompt_encoder' in k]
        prompt_encoder_values = [state_dict[k] for k in prompt_encoder_keys]
        prompt_encoder_new_state_dict = {k: v for k, v in zip(prompt_encoder_keys, prompt_encoder_values)}
        sam_dict.update(prompt_encoder_new_state_dict)

        # load mask decoder
        mask_decoder_keys = [k for k in sam_keys if 'mask_decoder' in k]
        mask_decoder_values = [state_dict[k] for k in mask_decoder_keys]
        mask_decoder_new_state_dict = {k: v for k, v in zip(mask_decoder_keys, mask_decoder_values)}
        sam_dict.update(mask_decoder_new_state_dict)

        # load prompt encoder
        prompt_encoder2_keys = [k for k in sam_keys if 'prompt_encoder2' in k]
        prompt_encoder2_values = [state_dict[k] for k in prompt_encoder2_keys]
        prompt_encoder2_new_state_dict = {k: v for k, v in zip(prompt_encoder2_keys, prompt_encoder2_values)}
        sam_dict.update(prompt_encoder2_new_state_dict)

        # load mask decoder
        mask_decoder2_keys = [k for k in sam_keys if 'mask_decoder2' in k]
        mask_decoder2_values = [state_dict[k] for k in mask_decoder2_keys]
        mask_decoder2_new_state_dict = {k: v for k, v in zip(mask_decoder2_keys, mask_decoder2_values)}
        sam_dict.update(mask_decoder2_new_state_dict)

        self.sam.load_state_dict(sam_dict)

    def reset_parameters(self) -> None:
        for w_A in self.w_As:
            nn.init.kaiming_uniform_(w_A.weight, a=math.sqrt(5))
        for w_B in self.w_Bs:
            nn.init.zeros_(w_B.weight)

    def forward(self, batched_input, multimask_output =True, mode = "train", pre_mask = None, previous_mask = None):
        return self.sam(batched_input, multimask_output, pre_mask = pre_mask, previous_mask = previous_mask)



if __name__ == "__main__":
    sam = sam_model_registry["vit_b"](checkpoint="sam_vit_b_01ec64.pth")
    lora_sam = LoRA_Sam(sam, 4)
    lora_sam.sam.image_encoder(torch.rand(size=(1, 3, 1024, 1024)))
