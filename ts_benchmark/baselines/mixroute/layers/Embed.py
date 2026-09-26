import math
import torch
import torch.nn as nn


class PositionalEmbedding(nn.Module):
    

    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float()
                    * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]


def patch_geometry(seq_len, patch_len):
    
    seq_len, patch_len = int(seq_len), int(patch_len)
    assert patch_len >= 1, f'patch_len must be >= 1, got {patch_len}'
    assert seq_len >= patch_len, f'seq_len({seq_len}) must be >= patch_len({patch_len})'
    n = seq_len // patch_len
    return n, n * patch_len


class PatchEmbedding(nn.Module):
    

    def __init__(self, d_model, patch_len, dropout):
        super(PatchEmbedding, self).__init__()
        self.patch_len = patch_len

        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        
        _, n_vars, input_len = x.shape
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.patch_len)
        x = torch.reshape(x, (x.shape[0] * x.shape[1], x.shape[2], x.shape[3]))
        x = self.value_embedding(x)
        x = x + self.position_embedding(x)
        return self.dropout(x), n_vars


class PerVariableInvertedEmbedding(nn.Module):
    

    def __init__(self, c_in, num_vars, d_model, dropout=0.1):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_vars, d_model, c_in))
        self.bias = nn.Parameter(torch.empty(num_vars, d_model))
        for i in range(num_vars):
            nn.init.kaiming_uniform_(self.weight[i], a=math.sqrt(5))
        bound = 1 / math.sqrt(c_in)
        nn.init.uniform_(self.bias, -bound, bound)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x, x_mark=None):
        
        x = x.permute(0, 2, 1)
        out = torch.einsum('bvt,vdt->bvd', x, self.weight) + self.bias
        return self.dropout(out)
