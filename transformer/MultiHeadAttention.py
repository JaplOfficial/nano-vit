import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        E_q: int,
        E_k: int,
        E_v: int,
        E_total: int,
        nheads: int,
        dropout: float = 0.0,
        bias = True,
        device = None,
        dtype = None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.nheads = nheads
        self.dropout = dropout
        self._qvk_same_embed_dim = E_q == E_k and E_k == E_v
        if self._qvk_same_embed_dim:
            self.packed_proj = nn.Linear(E_q, E_total * 3, bias=bias, **factory_kwargs)
        else:
            self.q_proj = nn.Linear(E_q, E_total, bias=bias, **factory_kwargs)  
            self.k_proj = nn.Linear(E_k, E_total, bias=bias, **factory_kwargs)  
            self.v_proj = nn.Linear(E_v, E_total, bias=bias, **factory_kwargs)
        E_out = E_q
        self.out_proj = nn.Linear(E_total, E_out, bias=bias, **factory_kwargs)
        assert E_total % nheads == 0, "Embedding dim is not divisible by nheads"
        self.E_head = E_total // nheads
        self.bias = bias

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask = None,
        is_causal = False,
    ) -> torch.Tensor:
        if self._qvk_same_embed_dim:
            if query is key and key is value:
                result = self.packed_proj(query)
                query, key, value = torch.chunk(result, 3, dim = -1)
            else:
                q_weight, k_weight, v_weight = torch.chunk(self.packed_proj.weight, 3, dim = 0)
                if self.bias:
                    q_bias, k_bias, v_bias = torch.chunk(self.packed_proj.bias, 3, dim = 0)
                else:
                    q_bias, k_bias, v_bias = None, None, None
                    query, key, value = (
                        F.linear(query, q_weight, q_bias),
                        F.linear(key, k_weight, k_bias),
                        F.linear(value, v_weight, v_bias)
                    )
        else:
            query = self.q_proj(query)
            key = self.k_proj(key)
            value = self.v_proj(value)
        
        #Magia
        query = query.unflatten(-1, (self.nheads, self.E_head)).transpose(1, 2)
        key = key.unflatten(-1, (self.nheads, self.E_head)).transpose(1, 2)
        value = value.unflatten(-1, (self.nheads, self.E_head)).transpose(1, 2)

        attn_output = F.scaled_dot_product_attention(
            query, key, value, dropout_p = self.dropout, is_causal = is_causal
        )
        attn_output = attn_output.transpose(1, 2).flatten(-2)
        attn_output = self.out_proj(attn_output)

        return attn_output

