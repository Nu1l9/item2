from torch import nn
import layernorm
import tools

class DecoderLayer(nn.Module):
    def __init__(self, size, attn, feed_forward, sublayer_num, dropout = 0.1):
        super(DecoderLayer,self).__init__()
        self.attn = attn
        self.feed_forward = feed_forward
        self.sublayer_connection = tools.clones(layernorm.SubLayerNorm(size, dropout), sublayer_num)

    def forward(self, x, memory,src_mask, trg_mask, r2l_memory=None, r2l_trg_mask = None):
        x = self.sublayer_connection[0](x ,lambda x:self.attn(x, x, x, trg_mask))#一层自注意力：x与x交互
        x = self.sublayer_connection[1](x ,lambda x:self.attn(x, memory, memory, src_mask))#二层注意力：x与memory进行的交互

        if r2l_memory is not None:#如果需要右向左的注意力机制 即双向解码器
            x = self.sublayer_connection[-2](x ,lambda x:self.attn(x, r2l_memory, r2l_memory, r2l_trg_mask))#倒数第二层的计算x与反向的r2l_memory
        return self.sublayer_connection[-1](x, self.feed_forward)

class R2L_Decoder(nn.Module):
    """
    model:反向解码器
    params: n: n层
    params: decoder_layer: 其中一层decoder
    """
    def __init__(self, n, decoder_layer):
        super(R2L_Decoder, self).__init__()
        self.decoder_layer = tools.clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, r2l_trg_mask)
        return x

class L2R_Decoder(nn.Module):
    """
    model:正向解码器
    params: n: n层
    params: decoder_layer: 其中一层decoder
    """

    def __init__(self, n, decoder_layer):
        super(L2R_Decoder, self).__init__()
        self.decoder_layer = clones(decoder_layer, n)

    def forward(self, x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask):
        for layer in self.decoder_layer:
            x = layer(x, memory, src_mask, trg_mask, r2l_memory, r2l_trg_mask)
        return x


