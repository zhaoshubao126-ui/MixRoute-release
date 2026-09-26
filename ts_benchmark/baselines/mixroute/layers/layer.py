import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.activations import ACT2FN
from typing import Optional, Tuple


def rotate_half(x):
    
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim=1):
    
    cos = cos[position_ids].unsqueeze(unsqueeze_dim)
    sin = sin[position_ids].unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class RotaryEmbedding(torch.nn.Module):
    

    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2, dtype=torch.int64).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device, dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=torch.int64).type_as(self.inv_freq)
        freqs = torch.outer(t, self.inv_freq)
        
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().to(dtype), persistent=False)
        self.register_buffer("sin_cached", emb.sin().to(dtype), persistent=False)

    def forward(self, x, seq_len=None):
        
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device, dtype=x.dtype)
        return (
            self.cos_cached[:seq_len].to(dtype=x.dtype),
            self.sin_cached[:seq_len].to(dtype=x.dtype),
        )


class TemporalBlock(nn.Module):
    

    def __init__(self, hidden_size: int, intermediate_size: int, hidden_act: str):
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, hidden_state):
        return self.down_proj(self.act_fn(self.gate_proj(hidden_state)) * self.up_proj(hidden_state))


class MixedFFN(nn.Module):
    

    def __init__(self,
                 hidden_size: int,
                 intermediate_size: int,
                 hidden_act: str,
                 num_endo_tokens: int,
                 num_exog_tokens: int,
                 num_glob_tokens: int):
        super().__init__()

        self.num_endo_tokens = num_endo_tokens
        self.num_exog_tokens = num_exog_tokens
        self.num_glob_tokens = num_glob_tokens
        self.total_tokens = num_endo_tokens + num_exog_tokens + num_glob_tokens

        
        self.shared_ffn = TemporalBlock(hidden_size=hidden_size, intermediate_size=intermediate_size, hidden_act=hidden_act)

        
        num_ns = num_exog_tokens + num_glob_tokens
        self.ns_gate_weight = nn.Parameter(torch.empty(num_ns, intermediate_size, hidden_size))
        self.ns_up_weight = nn.Parameter(torch.empty(num_ns, intermediate_size, hidden_size))
        self.ns_down_weight = nn.Parameter(torch.empty(num_ns, hidden_size, intermediate_size))

        self.act_fn = ACT2FN[hidden_act]

        self.reset_parameters()

    def reset_parameters(self):
        num_ns = self.num_exog_tokens + self.num_glob_tokens
        if num_ns > 0:
            for i in range(num_ns):
                nn.init.kaiming_uniform_(self.ns_gate_weight[i], a=5 ** 0.5)
                nn.init.kaiming_uniform_(self.ns_up_weight[i], a=5 ** 0.5)
                nn.init.kaiming_uniform_(self.ns_down_weight[i], a=5 ** 0.5)

    def forward(self, hidden_state: torch.Tensor):
        
        if hidden_state.dim() != 3:
            raise ValueError(
                f"hidden_state should be [B, L, H], got {hidden_state.shape}"
            )
        B, L, H = hidden_state.shape
        if L != self.total_tokens:
            raise ValueError(
                f"Token length mismatch: expected {self.total_tokens}, got {L}"
            )

        seq_hidden = hidden_state[:, :self.num_endo_tokens, :]
        ns_hidden = hidden_state[:, self.num_endo_tokens:, :]

        
        outputs_s = self.shared_ffn(seq_hidden)  

        
        gate_out = torch.einsum("bnh,nih->bni", ns_hidden, self.ns_gate_weight)
        up_out = torch.einsum("bnh,nih->bni", ns_hidden, self.ns_up_weight)
        gated = self.act_fn(gate_out) * up_out
        outputs_ns = torch.einsum("bni,nhi->bnh", gated, self.ns_down_weight)

        return torch.cat([outputs_s, outputs_ns], dim=1)  


class MixedAttention(nn.Module):
    

    def __init__(self, config,
                 layer_idx: Optional[int] = None,
                 num_endo_tokens: int = None,
                 num_exog_tokens: int = None,
                 num_glob_tokens: int = None):
        super().__init__()
        self.num_endo_tokens = num_endo_tokens
        self.num_exog_tokens = num_exog_tokens
        self.num_glob_tokens = num_glob_tokens
        self.total_tokens = self.num_endo_tokens + self.num_exog_tokens + self.num_glob_tokens

        self.hidden_size = config.d_model
        self.num_heads = config.n_heads
        self.head_dim = self.hidden_size // self.num_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta
        self.attention_dropout = config.dropout

        if (self.head_dim * self.num_heads) != self.hidden_size:
            raise ValueError(
                f"hidden_size must be divisible by num_heads (got `hidden_size`: {self.hidden_size}"
                f" and `num_heads`: {self.num_heads})."
            )

        
        self.qkv_bias = getattr(config, 'qkv_bias', True)
        self.q_proj_s = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)
        self.k_proj_s = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)
        self.v_proj_s = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)

        
        num_ns = num_exog_tokens + num_glob_tokens
        self.q_proj_ns = nn.Parameter(torch.empty(num_ns, self.hidden_size, self.hidden_size))
        self.k_proj_ns = nn.Parameter(torch.empty(num_ns, self.hidden_size, self.hidden_size))
        self.v_proj_ns = nn.Parameter(torch.empty(num_ns, self.hidden_size, self.hidden_size))
        self.q_bias_ns = nn.Parameter(torch.zeros(num_ns, self.hidden_size))
        self.k_bias_ns = nn.Parameter(torch.zeros(num_ns, self.hidden_size))
        self.v_bias_ns = nn.Parameter(torch.zeros(num_ns, self.hidden_size))

        
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)

        
        self.rotary_emb = RotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

        
        
        self.q_proj_int = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)
        
        self.q_proj_cross = nn.Linear(self.hidden_size, self.num_heads * self.head_dim, bias=self.qkv_bias)

        
        
        
        gate_input_dim = self.num_heads * self.head_dim * 2
        self.air_gate_proj = nn.Sequential(
            nn.LayerNorm(gate_input_dim),
            nn.Linear(gate_input_dim, self.num_heads),
            nn.Sigmoid(),
        )
        with torch.no_grad():
            self.air_gate_proj[1].bias.fill_(-2.0)

        
        intrinsic_mask = torch.triu(
            torch.full((self.num_endo_tokens, self.num_endo_tokens), float('-inf')),
            diagonal=1
        )
        self.register_buffer('air_causal_mask', intrinsic_mask)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj_s.weight)
        nn.init.xavier_uniform_(self.k_proj_s.weight)
        nn.init.xavier_uniform_(self.v_proj_s.weight)
        if self.qkv_bias:
            nn.init.zeros_(self.q_proj_s.bias)
            nn.init.zeros_(self.k_proj_s.bias)
            nn.init.zeros_(self.v_proj_s.bias)
        nn.init.normal_(self.q_proj_ns, std=0.02)
        nn.init.normal_(self.k_proj_ns, std=0.02)
        nn.init.normal_(self.v_proj_ns, std=0.02)
        nn.init.zeros_(self.q_bias_ns)
        nn.init.zeros_(self.k_bias_ns)
        nn.init.zeros_(self.v_bias_ns)
        nn.init.xavier_uniform_(self.o_proj.weight)

        
        
        self.q_proj_int.weight.data.copy_(self.q_proj_s.weight.data)
        if self.qkv_bias:
            self.q_proj_int.bias.data.copy_(self.q_proj_s.bias.data)
        nn.init.xavier_uniform_(self.q_proj_cross.weight)
        if self.qkv_bias:
            nn.init.zeros_(self.q_proj_cross.bias)

    def _shape_to_heads(self, x: torch.Tensor, num_tokens: int) -> torch.Tensor:
        
        B, T, D = x.shape
        if T != num_tokens:
            raise ValueError(f"Token length mismatch: expected {num_tokens}, got {T}")
        return x.view(B, T, self.num_heads, self.head_dim).transpose(1, 2).contiguous()

    def _project_ns_tokens(self, x_ns: torch.Tensor):
        
        B, NS, D = x_ns.shape
        expected_ns = self.num_exog_tokens + self.num_glob_tokens
        if NS != expected_ns:
            raise ValueError(f"NS token length mismatch: expected {expected_ns}, got {NS}")

        q_ns = torch.einsum("bni,noi->bno", x_ns, self.q_proj_ns) + self.q_bias_ns.unsqueeze(0)
        k_ns = torch.einsum("bni,noi->bno", x_ns, self.k_proj_ns) + self.k_bias_ns.unsqueeze(0)
        v_ns = torch.einsum("bni,noi->bno", x_ns, self.v_proj_ns) + self.v_bias_ns.unsqueeze(0)

        q_ns = self._shape_to_heads(q_ns, NS)
        k_ns = self._shape_to_heads(k_ns, NS)
        v_ns = self._shape_to_heads(v_ns, NS)
        return q_ns, k_ns, v_ns

    def _project_s_tokens(self, x_s: torch.Tensor, position_ids_s: Optional[torch.LongTensor]):
        
        B, S, D = x_s.shape

        q_s = self.q_proj_s(x_s)
        k_s = self.k_proj_s(x_s)
        v_s = self.v_proj_s(x_s)

        q_s = self._shape_to_heads(q_s, S)
        k_s = self._shape_to_heads(k_s, S)
        v_s = self._shape_to_heads(v_s, S)

        
        cos, sin = self.rotary_emb(v_s, seq_len=k_s.shape[-2])
        q_s, k_s = apply_rotary_pos_emb(q_s, k_s, cos, sin, position_ids_s)

        return q_s, k_s, v_s

    def forward(
            self,
            hidden_states: torch.Tensor,  
            position_ids: Optional[torch.LongTensor] = None,
            output_attentions: bool = False,
    ) -> Tuple:
        batch_size, seq_len, hidden_dim = hidden_states.shape
        if hidden_dim != self.hidden_size:
            raise ValueError(
                f"hidden_size mismatch: expected {self.hidden_size}, got {hidden_dim}"
            )
        if seq_len != self.total_tokens:
            raise ValueError(
                f"Input sequence length mismatch: expected {self.total_tokens}, got {seq_len}"
            )

        
        x_s = hidden_states[:, :self.num_endo_tokens, :]
        x_ns = hidden_states[:, self.num_endo_tokens:, :]

        if position_ids is not None:
            if position_ids.dim() != 2:
                raise ValueError(f"position_ids must be [B, T], got {position_ids.shape}")
            if position_ids.size(0) != batch_size:
                raise ValueError(
                    f"position_ids batch mismatch: expected {batch_size}, got {position_ids.size(0)}"
                )
            if position_ids.size(1) == self.total_tokens:
                position_ids_s = position_ids[:, :self.num_endo_tokens]
            elif position_ids.size(1) == self.num_endo_tokens:
                position_ids_s = position_ids
            else:
                raise ValueError(
                    f"position_ids length must be either total_tokens={self.total_tokens} "
                    f"or num_seq_tokens={self.num_endo_tokens}, got {position_ids.size(1)}"
                )
        else:
            position_ids_s = None

        
        q_s, k_s, v_s = self._project_s_tokens(x_s, position_ids_s)

        
        num_ns = self.num_exog_tokens + self.num_glob_tokens
        q_ns, k_ns, v_ns = self._project_ns_tokens(x_ns)

        
        
        q_int = self._shape_to_heads(self.q_proj_int(x_s), self.num_endo_tokens)
        q_cross = self._shape_to_heads(self.q_proj_cross(x_s), self.num_endo_tokens)

        cos, sin = self.rotary_emb(v_s, seq_len=self.num_endo_tokens)
        if position_ids_s is not None:
            cos_pos = cos[position_ids_s].unsqueeze(1)
            sin_pos = sin[position_ids_s].unsqueeze(1)
        else:
            cos_pos = cos.unsqueeze(0).unsqueeze(0)
            sin_pos = sin.unsqueeze(0).unsqueeze(0)
        q_int = (q_int * cos_pos) + (rotate_half(q_int) * sin_pos)

        
        score_int = torch.matmul(q_int, k_s.transpose(-2, -1)) / math.sqrt(self.head_dim)
        score_int = score_int + self.air_causal_mask.unsqueeze(0).unsqueeze(0)
        attn_int = F.softmax(score_int, dim=-1, dtype=torch.float32).to(q_int.dtype)
        attn_int = F.dropout(attn_int, p=self.attention_dropout, training=self.training)
        out_int = torch.matmul(attn_int, v_s)

        
        score_cross = torch.matmul(q_cross, k_ns.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_cross = F.softmax(score_cross, dim=-1, dtype=torch.float32).to(q_cross.dtype)
        attn_cross = F.dropout(attn_cross, p=self.attention_dropout, training=self.training)
        out_cross = torch.matmul(attn_cross, v_ns)

        
        B_a, H_a, _, d_h = out_int.shape
        int_feat = out_int.transpose(1, 2).reshape(B_a, -1, H_a * d_h).mean(dim=1)
        cross_feat = out_cross.transpose(1, 2).reshape(B_a, -1, H_a * d_h).mean(dim=1)
        gate_input = torch.cat([int_feat, cross_feat], dim=-1)
        gate = self.air_gate_proj(gate_input)
        gate = gate.view(batch_size, self.num_heads, 1, 1)

        if self.training:
            
            self._last_out_int = out_int
            self._last_out_cross = out_cross

        endo_out = out_int + gate * out_cross
        endo_out = endo_out.transpose(1, 2).contiguous().view(
            batch_size, self.num_endo_tokens, self.hidden_size)

        
        k_all = torch.cat([k_s, k_ns], dim=2)  
        v_all = torch.cat([v_s, v_ns], dim=2)
        attn_ns = torch.matmul(q_ns, k_all.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn_ns = F.softmax(attn_ns, dim=-1, dtype=torch.float32).to(q_ns.dtype)
        attn_ns = F.dropout(attn_ns, p=self.attention_dropout, training=self.training)
        ns_out = torch.matmul(attn_ns, v_all)  
        ns_out = ns_out.transpose(1, 2).contiguous().view(
            batch_size, num_ns, self.hidden_size)

        
        attn_output = torch.cat([endo_out, ns_out], dim=1)
        attn_output = self.o_proj(attn_output)

        if output_attentions:
            attn_weights = {
                'intrinsic': attn_int.detach(),
                'cross_type': attn_cross.detach(),
                'ns': attn_ns.detach(),
            }
        else:
            attn_weights = None
        return attn_output, attn_weights


class DropPath(nn.Module):
    

    def __init__(self, drop_prob: float = 0.0, scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        batch_size = x.shape[0]
        
        random_tensor = torch.rand(batch_size, 1, 1, device=x.device, dtype=x.dtype)
        random_tensor += keep_prob
        binary_mask = torch.floor(random_tensor)
        if self.scale_by_keep:
            output = x / keep_prob * binary_mask
        else:
            output = x * binary_mask
        return output

    def extra_repr(self):
        return f'drop_prob={self.drop_prob:.3f}, scale_by_keep={self.scale_by_keep}'


class RMSNorm(torch.nn.Module):
    

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)
