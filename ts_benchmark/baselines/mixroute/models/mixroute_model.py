

import torch
import torch.nn.functional as F
from torch import nn

from ts_benchmark.baselines.mixroute.layers.layer import (
    MixedAttention,
    MixedFFN,
    RMSNorm,
    DropPath,
)
from ts_benchmark.baselines.mixroute.layers.Embed import (
    PatchEmbedding,
    PerVariableInvertedEmbedding,
    patch_geometry,
)


class FlattenHead(nn.Module):
    

    def __init__(self, num_endo_tokens, num_glob_tokens,
                 d_model, d_ff, pred_len, endo_num, head_dropout):
        super().__init__()
        self.pred_len = pred_len
        self.endo_num = endo_num

        self.glob_dim = num_glob_tokens * d_model
        self.glob_flatten = nn.Flatten(start_dim=1)
        self.mlp = nn.Sequential(
            nn.Linear(self.glob_dim, d_ff),
            nn.GELU(),
            nn.Dropout(head_dropout),
            nn.Linear(d_ff, pred_len * endo_num),
        )

        
        self.temporal_proj = nn.Linear(num_endo_tokens, pred_len)
        self.channel_proj = nn.Linear(d_model, endo_num)
        self.residual_gate = nn.Parameter(torch.tensor(0.1))

        self.dropout = nn.Dropout(head_dropout)

    def forward(self, endo_states, glob_states):
        
        B = endo_states.size(0)

        pred = self.mlp(self.glob_flatten(glob_states))  

        endo_res = self.temporal_proj(endo_states.transpose(1, 2))
        endo_res = self.channel_proj(endo_res.transpose(1, 2))
        endo_res = endo_res.reshape(B, -1)
        pred = pred + self.residual_gate * endo_res

        pred = self.dropout(pred)
        return pred.view(B, self.pred_len, self.endo_num)


class EncoderLayer(nn.Module):
    

    def __init__(self, config, layer_idx, num_endo_tokens, num_exog_tokens, num_glob_tokens):
        super().__init__()
        self.num_layers = config.e_layers
        self.total_tokens = num_endo_tokens + num_exog_tokens + num_glob_tokens

        self.ffn_layer = MixedFFN(
            hidden_size=config.d_model,
            intermediate_size=config.d_ff,
            hidden_act=config.hidden_act,
            num_endo_tokens=num_endo_tokens,
            num_exog_tokens=num_exog_tokens,
            num_glob_tokens=num_glob_tokens,
        )
        self.self_attn = MixedAttention(
            config=config,
            layer_idx=layer_idx,
            num_endo_tokens=num_endo_tokens,
            num_exog_tokens=num_exog_tokens,
            num_glob_tokens=num_glob_tokens,
        )
        self.input_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.post_layernorm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.ffn_dropout = nn.Dropout(config.dropout)

        
        drop_path_rate = config.drop_path_rate
        if drop_path_rate > 0:
            self.drop_path = DropPath(
                drop_prob=drop_path_rate * layer_idx / max(self.num_layers - 1, 1)
            )
        else:
            self.drop_path = nn.Identity()

    def forward(self, hidden_states):
        if hidden_states.shape[1] != self.total_tokens:
            raise ValueError(
                f"Token length mismatch: expected {self.total_tokens}, "
                f"got {hidden_states.shape[1]}"
            )

        residual = hidden_states
        attn_out, _ = self.self_attn(self.input_layernorm(hidden_states))
        hidden_states = residual + self.drop_path(self.attn_dropout(attn_out))

        residual = hidden_states
        ffn_out = self.ffn_layer(self.post_layernorm(hidden_states))
        hidden_states = residual + self.drop_path(self.ffn_dropout(ffn_out))

        return hidden_states


class MixRouteModel(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.d_model = config.d_model
        self.num_layers = config.e_layers
        self.pred_len = config.pred_len
        self.patch_len = config.patch_len
        self.use_norm = getattr(config, 'use_instance_norm', True)

        self.num_endo_vars = config.series_dim
        self.num_exog_vars = config.enc_in - config.series_dim
        self.num_glob_tokens = config.num_glob_tokens

        
        self.num_endo_tokens, self.input_len = patch_geometry(config.seq_len, config.patch_len)
        self.num_exog_tokens = self.num_exog_vars

        self.endo_embedding = PatchEmbedding(
            d_model=self.d_model, patch_len=self.patch_len,
            dropout=config.dropout)
        
        self.exog_embedding = PerVariableInvertedEmbedding(
            c_in=self.input_len + self.pred_len, num_vars=self.num_exog_vars,
            d_model=self.d_model, dropout=config.dropout)

        self.glob_tokens = nn.Parameter(torch.empty(self.num_glob_tokens, self.d_model))
        nn.init.normal_(self.glob_tokens, std=0.02)

        self.type_embedding = nn.Embedding(3, self.d_model)
        nn.init.normal_(self.type_embedding.weight, std=1.0)
        type_ids = torch.cat([
            torch.zeros(self.num_endo_tokens, dtype=torch.long),
            torch.ones(self.num_exog_tokens, dtype=torch.long),
            torch.full((self.num_glob_tokens,), 2, dtype=torch.long),
        ])
        self.register_buffer('type_ids', type_ids)

        self.encoder = nn.ModuleList([
            EncoderLayer(config=config, layer_idx=i,
                         num_endo_tokens=self.num_endo_tokens,
                         num_exog_tokens=self.num_exog_tokens,
                         num_glob_tokens=self.num_glob_tokens)
            for i in range(self.num_layers)
        ])

        self.head = FlattenHead(
            num_endo_tokens=self.num_endo_tokens,
            num_glob_tokens=self.num_glob_tokens,
            d_model=self.d_model,
            d_ff=config.d_ff,
            pred_len=self.pred_len,
            endo_num=self.num_endo_vars,
            head_dropout=config.dropout,
        )

        self.air_orth_weight = getattr(config, 'air_orth_weight', 0.0)

    def sample_norm(self, x, means, stdev):
        return (x - means) / stdev

    def sample_denorm(self, x, means, stdev):
        seq_len = x.shape[1]
        x = x * (stdev[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        x = x + (means[:, 0, :].unsqueeze(1).repeat(1, seq_len, 1))
        return x

    def compute_air_orthogonal_loss(self):
        
        losses = []
        for layer in self.encoder:
            attn = layer.self_attn
            if getattr(attn, '_last_out_int', None) is not None\
                    and getattr(attn, '_last_out_cross', None) is not None:
                
                h_int = attn._last_out_int.mean(dim=2).reshape(attn._last_out_int.shape[0], -1)
                h_cross = attn._last_out_cross.mean(dim=2).reshape(attn._last_out_cross.shape[0], -1)
                losses.append(F.cosine_similarity(h_int, h_cross, dim=-1).abs().mean())
        if losses:
            return self.air_orth_weight * torch.stack(losses).mean()
        return torch.tensor(0.0, device=next(self.parameters()).device)

    def forward(self, history, exog_future):
        endo_history = history[:, -self.input_len:, :self.num_endo_vars]
        exog_history = history[:, -self.input_len:, self.num_endo_vars:]
        exog_future = exog_future[:, :self.pred_len, :]

        if self.use_norm:
            endo_means = endo_history.mean(1, keepdim=True).detach()
            endo_stdev = torch.sqrt(
                torch.var(endo_history, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            exog = torch.cat([exog_history, exog_future], dim=1)
            
            exog_means = exog.mean(1, keepdim=True).detach()
            exog_stdev = torch.sqrt(
                torch.var(exog, dim=1, keepdim=True, unbiased=False) + 1e-5).detach()

            endo_history = self.sample_norm(endo_history, endo_means, endo_stdev)
            exog = self.sample_norm(exog, exog_means, exog_stdev)

        endo_embed, _ = self.endo_embedding(endo_history.permute(0, 2, 1))  
        exog_embed = self.exog_embedding(exog, None)                        
        glob_embed = self.glob_tokens.unsqueeze(0).expand(endo_embed.shape[0], -1, -1)

        hidden_states = torch.cat([endo_embed, exog_embed, glob_embed], dim=1)
        hidden_states = hidden_states + self.type_embedding(self.type_ids)

        for encoder_layer in self.encoder:
            hidden_states = encoder_layer(hidden_states)

        endo_end = self.num_endo_tokens
        exog_end = endo_end + self.num_exog_tokens
        endo_states = hidden_states[:, :endo_end, :]
        glob_states = hidden_states[:, exog_end:, :]

        pred = self.head(endo_states, glob_states)

        additional_loss = None
        if self.use_norm:
            pred = self.sample_denorm(pred, endo_means, endo_stdev)

        if self.training and self.air_orth_weight > 0:
            additional_loss = self.compute_air_orthogonal_loss()

        return pred, additional_loss
