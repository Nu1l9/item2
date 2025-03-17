import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn import LayerNorm

class Layernorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        """
        :param features:self-attention的 x 大小
        """
        super(Layernorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps
    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return (x-mean)/(std+self.eps) * self.gamma + self.beta



class SubLayerNorm(nn.Module):

    def __init__(self, size, dropout=0.1):
        ##同时完成残差和Layernorm的计算
        super(SubLayerNorm, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        """
        :param x:上一层self-attention的输入
        :param sublayer:上一层attention
        :return:
        """
        return self.dropout(self.layer_norm(x+sublayer(x)))##顺道完成了残差