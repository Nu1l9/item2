from torch import nn


class PositionalFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionalFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.lay_norm = nn.LayerNorm(d_model,eps=1e-6)
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.ReLU = nn.ReLU()

    def forward(self,x):
        inter = self.dropout_1(self.ReLU(self.w_1(self.lay_norm(x))))
        out = self.dropout_2(self.w_2(inter))
        return out
