import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from networkx.generators.degree_seq import directed_configuration_model
from torch.ao.nn.quantized.functional import linear


class Token_Embedding(nn.Embedding):
    def __init__(self, vocab_size, d_model):#vocab_size代表词表大小，而d_model代表了所需要的维度
        super(Token_Embedding, self).__init__(vocab_size, d_model, padding_idx=0)

class LayerNorm(nn.Module):

    def __init__(self, feature, eps=1e-6):
        """
        :param feature: self-attention 的 x 的大小
        :param eps:
        """
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps

    def forward(self, x):
        mean = x.float().mean(-1, keepdim=True)
        std = x.float().std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

class Positional_Embedding(nn.Module):
    def __init__(self, dim,dropout,max_len=5000):
        super(Positional_Embedding, self).__init__()
        if dim%2 != 0:
            raise ValueError("cannot use sin/cos positional embedding")

        pe = torch.zeros(max_len,dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))

        pe[:,0::2] = torch.sin(position/div_term)
        pe[:,1::2] = torch.cos(position/div_term)
        pe = pe.unsqueeze(1)
        self.register_buffer('pe', pe)
        self.dropout = nn.Dropout(dropout)
        self.dim = dim


    def forward(self, emb, step):
        if step is not None:
            emb = emb + self.pe[:,emb.size(0)]
        else:
            emb = emb +self.pe[step]
        emb = self.dropout(emb)

        return emb

class Transformer_Embedding(nn.Module):
    def __init__(self,vocab_size, d_model, max_len, drop_prob, device):
        super(Transformer_Embedding, self).__init__()
        self.tok_emb = Token_Embedding(vocab_size, d_model)
        self.pos_emb = Positional_Embedding(d_model, max_len, device)
        self.drop = nn.Dropout(p = drop_prob)

    def forward(self,x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.drop_out(tok_emb+pos_emb)

class Feat_Embedding(nn.Module):

    def __init__(self, d_feat, d_model, dropout):
        super(Feat_Embedding, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(d_feat),
            nn.Dropout(dropout),
            nn.Linear(d_feat, d_model))

    def forward(self, x):
        return self.video_embeddings(x)

class Text_Embedding(nn.Module):

    def __init__(self, vocab_size, d_model):
        super(Text_Embedding, self).__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(vocab_size, d_model)

    def forward(self, x):
        return self.embed(x) * math.sqrt(self.d_model)