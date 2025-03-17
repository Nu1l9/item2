import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(self, head, d_model, dropout=0.1,mask = None):
        """
        :param d_model:输入的维度
        :head:头数
        :q : query
        :k : key
        :v : value
        :param dropout:丢弃比例
        :param mask:是否掩码
        """
        super(MultiHeadAttention, self).__init__()

        self.d_k = d_model // head
        self.head = head
        self.d_model = d_model
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)
        self.linear_out = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(p = dropout)
        self.attn = None
        if mask is not None:
            mask = mask.unsqueeze(1)

        def forward(x):
            n_batch = q.size(0)

            # 多头需要对这个 x 切分成多个头

            query = self.linear_q(q).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
            key = self.linear_k(k).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)
            value = self.linear_v(v).view(n_batch, -1, self.head, self.d_k).transpose(1, 2)

            x, self.attn = self_attention(query, key, value, dropout=self.dropout, mask=mask)
            x = x.transpose(1, 2).contiguous().view(n_batch, -1, self.head * self.d_k)
            return self.linear_out(x),self.attn






def self_attention(q, k, v,dropout=None, mask=None):
    ##自注意力公式
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
    self_attn = F.softmax(scores, dim=-1)

    if mask is not None:
        mask.cuda()
        scores = scores.masked_fill(mask == 0, -1e9)
    if dropout is not None:
        self_attn = dropout(self_attn)
    return torch.matmul(scores, v),self_attn


