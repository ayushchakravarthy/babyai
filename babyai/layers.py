import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable

def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout, max_len=64):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)], requires_grad=False)
        return self.dropout(x)

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2

def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(query, key, value, mask=mask,
        dropout=self.dropout)

        # 3) "Concat" using a view and apply a final linear.
        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)

def initialize_parameters(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.weight.data.normal_(0, 1)
        m.weight.data *= 1 / torch.sqrt(m.weight.data.pow(2).sum(1, keepdim=True))
        if m.bias is not None:
            m.bias.data.fill_(0)

class Encoder(nn.Module):
    def __init__(self,d_model,h,seq_len,in_channels,out_channels):
        super().__init__()
        self.position_encoding = PositionalEncoding(d_model,dropout=0.0)
        self.multi_head = MultiHeadedAttention(h,d_model,dropout=0.0)
        self.transition = nn.Linear(in_channels,out_channels)
        self.layer_norm = LayerNorm(torch.Size([d_model]))
        self.apply(initialize_parameters)

    def forward(self,x,t):
        x = self.position_encoding(x)
        for _ in range(t):
            mh = self.multi_head(x,x,x)
            x = x + mh
            norm = self.layer_norm(x)

            tran = self.transition(norm.squeeze(0))
            tran = tran + norm
            x = self.layer_norm(tran)

        return x

class Gate(nn.Module):
    def __init__(self,h,in_channels,out_channels):
        super().__init__()
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.multihead = MultiHeadedAttention(h,out_channels)
        # self.layer_norm = [
        #     nn.LayerNorm(torch.Size([8,8,out_channels])),
        #     nn.LayerNorm(torch.Size([4,4,out_channels])),
        #     nn.LayerNorm(torch.Size([2,2,out_channels])),
        # ]
        self.layer_norm = nn.LayerNorm(torch.Size([7,7,out_channels]))


    def forward(self,obs,instr,t):
        x = self.multihead(obs,instr,instr)
        x = x.view(x.size(0),obs.size(1),obs.size(2),x.size(-1))
        x = self.layer_norm(x)
        x = x + obs
        x = x.permute(0,3,1,2)
        x = self.transition(x)

        return x


class ReAttention(nn.Module):
    def __init__(self,h,in_channels,out_channels):
        super().__init__()
        self.attention = MultiHeadedAttention(h,out_channels)
        self.layer_norm = nn.LayerNorm(torch.Size([7,7,out_channels]))
        self.transition = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
            kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )
        self.linears = nn.ModuleList(
            [nn.Linear(2 * out_channels, out_channels) for _ in range(3)])


    def forward(self, outputs, enc_out):
        dec1 = outputs[0]
        dec2 = outputs[1]
        dec3 = outputs[2]

        dec1 = dec1.permute(0, 2, 3, 1)
        dec2 = dec2.permute(0, 2, 3, 1)
        dec3 = dec3.permute(0, 2, 3, 1)

        out = enc_out

        att1 = self.att(dec1, out, out)
        att2 = self.att(dec2, out, out)
        att3 = self.att(dec3, out, out)

        alpha1 = torch.sigmoid(self.linears[0](torch.cat([dec1, att1],-1)))
        alpha2 = torch.sigmoid(self.linears[1](torch.cat([dec2, att2],-1)))
        alpha3 = torch.sigmoid(self.linears[2](torch.cat([dec3, att3],-1)))

        outs = (alpha1*att1+alpha2*att2+alpha3*att3)/math.sqrt(3)
        outs = self.transition(outs.permute(0,3,1,2))

        return outs

    def att(self, query, key, value):
        att = self.attention(query, key, value)
        att = att.view(query.size(0), query.size(1), query.size(2), -1)
        out = self.layer_norm(att)

        out = out + query

        return out