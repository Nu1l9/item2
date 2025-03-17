from torch import nn

import layernorm
import tools

class EncoderLayer(nn.Module):
    def __init__(self, size, attn, feedforward, dropout = 0.1):
        super(EncoderLayer, self).__init__()

        self.attn = attn
        self.feedforward = feedforward
        self.sublayer_connection = tools.clones(layernorm.SubLayerNorm(size, dropout), 2)

    def forward(self, x, mask = None):
        x = self.sublayer_connection[0](x,lambda x:self.attn(x, x, x, mask))
        return self.sublayer_connection[1](x,self.feedforward)

class Encoder(nn.Module):
    def __init__(self, n, encoder_layer):
        super(Encoder, self).__init__()
        self.encoder_layer = tools.clones(encoder_layer, n)

    def forward(self, x, src_mask):
        for layer in self.encoder_layer:
            x = layer(x, src_mask)
        return x


