import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import copy

def clones(module, N):
   return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def pad_mask(src,r2l_trg,trg,pad_idx):
    if isinstance(src,tuple):
        if len(src)==4:
            src_image_mask = (src[0][:,:,0] != pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:,:,0]!= pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:,:,0]!= pad_idx).unsqueeze(1)
            src_rel_mask = (src[3][:, :, 0] != pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask,src_motion_mask,src_object_mask,src_rel_mask)
            dec_src_mask_1 = src_image_mask & src_motion_mask
            dec_src_mask_2 = src_image_mask & src_rel_mask & src_motion_mask & src_object_mask
            dec_src_mask = (dec_src_mask_1,dec_src_mask_2)
            src_mask = (enc_src_mask,dec_src_mask)
        elif len(src)==3:
            src_image_mask = (src[0][:,:,0]!= pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:,:,0]!= pad_idx).unsqueeze(1)
            src_object_mask = (src[2][:,:,0]!= pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask, src_object_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask,dec_src_mask)
        elif len(src)==2:
            src_image_mask = (src[0][:,:,0]!= pad_idx).unsqueeze(1)
            src_motion_mask = (src[1][:,:,0]!= pad_idx).unsqueeze(1)
            enc_src_mask = (src_image_mask, src_motion_mask)
            dec_src_mask = src_image_mask & src_motion_mask
            src_mask = (enc_src_mask,dec_src_mask)
    else:
        src_mask = (src[:,:,0]!= pad_idx).unsqueeze(1)

    if trg is not None:
        if isinstance(src_mask, tuple):
            trg_mask = (trg !=pad_idx).unsqueeze(1) & Mask.subsequent_mask(trg.size(1)).type_as(src_mask)
            r2l_pad_mask = (r2l_trg !=pad_idx).unsqueeze(1).type_as(src_image_mask)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_image_mask)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask

        else:
            trg_mask = (trg != pad_idx).unsqueeze(1) & Mask.subsequent_mask(trg.size(1)).type_as(src_mask)
            r2l_pad_mask = (r2l_trg !=pad_idx).unsqueeze(1).type_as(src_mask)
            r2l_trg_mask = r2l_pad_mask & subsequent_mask(r2l_trg.size(1)).type_as(src_mask)
            return src_mask, r2l_pad_mask, r2l_trg_mask, trg_mask
    else:
        return src_mask

def subsequent_mask(size):
    attn_shape = (1, size, size)
    mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return (torch.from_numpy(mask)).cuda


class Generator(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Generator, self).__init__()
        self.linear = nn.Linear(d_model, d_model)

    def forward(self, x):
        return F.log_softmax(self.linear(x), dim=-1)

