import torch
import torch.nn as nn
import copy

import attention
import Embedding
import layernorm
import Decoder
import Encoder
import feedforward
import tools


class Transformer(nn.Module):
    def __init__(self, vocab, d_feat, d_model, d_ff=2048, n_heads=6, n_layers=8, dropout=0.1, feature_mode ="one",
                 device='cuda', n_heads_big=128):
        super(Transformer, self).__init__()
        self.vocab = vocab
        self.device = device
        self.feature_mode = feature_mode
        self.pad_idx = vocab.get('<pad>', None)
        self.vocab_size = len(vocab)

        c = copy.deepcopy
        attn = attention.MultiHeadAttention(n_heads, d_model, dropout)
        feed_forward = feedforward.PositionalFeedForward(d_model, d_ff)


        if feature_mode == 'one':
            self.src_embed = Embedding.Feat_Embedding(d_feat, d_model, dropout)
        elif feature_mode == 'two':
            self.image_src_embed = Embedding.Feat_Embedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = Embedding.Feat_Embedding(d_feat[1], d_model, dropout)
        elif feature_mode == 'three':
            self.image_src_embed = Embedding.Feat_Embedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = Embedding.Feat_Embedding(d_feat[1], d_model, dropout)
            self.object_src_embed = Embedding.Feat_Embedding(d_feat[2], d_model, dropout)
        elif feature_mode == 'four':
            self.image_src_embed = Embedding.Feat_Embedding(d_feat[0], d_model, dropout)
            self.motion_src_embed = Embedding.Feat_Embedding(d_feat[1], d_model, dropout)
            self.object_src_embed = Embedding.Feat_Embedding(d_feat[2], d_model, dropout)
            self.rel_src_embed = Embedding.Feat_Embedding(d_feat[3], d_model, dropout)
        self.trg_embed = Embedding.Text_Embedding(len(vocab), d_model)
        self.pos_embed = Embedding.Positional_Embedding(d_model, dropout)

        self.encoder = Encoder.Encoder(n_layers, Encoder.EncoderLayer(d_model, c(attn), c(feed_forward), dropout))
        self.generator = tools.Generator(d_model,self.vocab_size)

    def encode(self, src, src_mask, feature_mode_two=False):
        if self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        if feature_mode_two:
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        if self.feature_mode == 'one':
            x = self.src_embed(src)
            x = self.pos_embed(x)
            return self.encoder(x, src_mask)
        elif self.feature_mode == 'two':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder_big(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder_big(x2, src_mask[1])
            return x1 + x2
        elif self.feature_mode == 'three':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])
            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])
            x3 = self.object_src_embed(src[2])
            x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, src_mask[2])
            return x1 + x2 + x3
        elif self.feature_mode == 'four':
            x1 = self.image_src_embed(src[0])
            x1 = self.pos_embed(x1)
            x1 = self.encoder(x1, src_mask[0])

            x2 = self.motion_src_embed(src[1])
            x2 = self.pos_embed(x2)
            x2 = self.encoder(x2, src_mask[1])

            x3 = self.object_src_embed(src[2])
            # x3 = self.pos_embed(x3)
            x3 = self.encoder(x3, src_mask[2])
            # x3 = self.encoder_no_attention(x3, src_mask[2])

            x4 = self.rel_src_embed(src[3])
            # x4 = self.pos_embed(x4)
            # x4 = self.encoder_no_
            # heads(x4, src_mask[3])
            x4 = self.encoder_no_attention(x4, src_mask[3])
            # x4 = self.encoder(x4, src_mask[3])
            return x1 + x2 + x3 + x4

    def r2l_decode(self, r2l_trg, memory, src_mask, r2l_trg_mask):
        x = self.trg_embed(r2l_trg)
        x = self.pos_embed(x)
        return self.r2l_decoder(x, memory, src_mask, r2l_trg_mask)

    def l2r_decode(self, trg, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        x = self.trg_embed(trg)
        x = self.pos_embed(x)
        return self.l2r_decoder(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)

    def forward(self, src, r2l_trg, trg, mask):
        print(f"Mask shape: {mask.shape}")  # 查看mask的形状
        src_mask = mask  # 如果mask已经是一个合适的张量，直接使用它
        r2l_pad_mask = mask
        r2l_trg_mask = mask
        trg_mask = mask

        if self.feature_mode == 'one':
            encoding_outputs = self.encode(src, src_mask)
            r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, src_mask, r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, src_mask, trg_mask, r2l_outputs, r2l_pad_mask)

        elif self.feature_mode == 'two' or 'three' or 'four':
            enc_src_mask = src_mask
            dec_src_mask = src_mask
            r2l_encoding_outputs = self.encode(src, enc_src_mask, feature_mode_two=True)
            encoding_outputs = self.encode(src, enc_src_mask)

            r2l_outputs = self.r2l_decode(r2l_trg, r2l_encoding_outputs, dec_src_mask[0], r2l_trg_mask)
            l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask[1], trg_mask, r2l_outputs, r2l_pad_mask)

            # r2l_outputs = self.r2l_decode(r2l_trg, encoding_outputs, dec_src_mask, r2l_trg_mask)
            # l2r_outputs = self.l2r_decode(trg, encoding_outputs, dec_src_mask, trg_mask, None, None)
        else:
            raise "没有输出"






