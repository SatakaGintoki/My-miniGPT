import torch
import torch.nn as nn
import torch.nn.init as init
import math
from einops import einsum
from einops import rearrange



def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")



class Linear(nn.Module):
    def __init__(self,in_features,out_features,device=None,dtype = None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), device=device, dtype=dtype)
        )

        std = math.sqrt(2.0 / (in_features + out_features))

        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = std,
            a = -3.0*std,
            b = 3.0*std
        )

    def forward(self,x):

        return einsum(self.weight,x,"d_out d_in , ... d_in -> ... d_out")



class Embedding(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()

        self.weight = nn.Parameter(
            torch.empty((num_embeddings, embedding_dim),device=device,dtype=dtype)
        )

        torch.nn.init.trunc_normal_(
            self.weight,
            mean = 0.0,
            std = 1.0,
            a = -3.0,
            b = 3.0
        )

    def forward(self,token_ids):
        return self.weight[token_ids]




class RMSNorm(nn.Module):

    def __init__(self, d_model: int, eps: float = 1e-5,device=None,dtype = None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps

        self.weight = nn.Parameter(
            torch.ones(d_model,device=device,dtype=dtype)
        )

    def forward(self,x):
        x_dtype = x.dtype
        x_fp32 = x.to(torch.float32)

        var = x_fp32.pow(2).mean(dim = -1,keepdim = True)
        rms = torch.sqrt(var+self.eps)

        x_norm = (x/rms).to(x.dtype)

        result = x_norm * self.weight

        return result




def SiLU(x):
    return x*torch.sigmoid(x)




class SwiGLU(nn.Module):
    def __init__(self,d_model,d_ff,device=None,dtype = None):
        super().__init__()

        if d_ff is None:
            d_ff = int(8/3*d_model)
            d_ff = ((d_ff+63)//64)*64
        self.d_ff = d_ff

        self.d_model = d_model

        factory_kwargs = {'device': device, 'dtype': dtype}

        self.w1 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        self.w3 = nn.Parameter(torch.empty((d_ff, d_model), **factory_kwargs))
        self.w2 = nn.Parameter(torch.empty((d_model, d_ff), **factory_kwargs))
        
        self.reset_parameters()

    def reset_parameters(self):
        std_in = (2.0 / (self.d_model + self.d_ff)) ** 0.5
        init.trunc_normal_(self.w1, mean=0.0, std=std_in, a=-3*std_in, b=3*std_in)
        init.trunc_normal_(self.w3, mean=0.0, std=std_in, a=-3*std_in, b=3*std_in)
        
        std_out = (2.0 / (self.d_ff + self.d_model)) ** 0.5
        init.trunc_normal_(self.w2, mean=0.0, std=std_out, a=-3*std_out, b=3*std_out)
        

    def forward(self,x):
        w1_x = einsum(x, self.w1,"... d_m , d_ff d_m -> ... d_ff")
        silu_w1_x = w1_x * torch.sigmoid(w1_x)

        w3_x = einsum( x, self.w3,"... d_m, d_ff d_m -> ... d_ff")

        w_x = silu_w1_x * w3_x

        output = einsum(w_x,self.w2,"... d_ff , d_m d_ff -> ... d_m")

        return output
        


class RoPE(nn.Module):

    def __init__(
        self, theta: float, d_k: int, max_seq_len: int, device=None
    ):
        super().__init__()
        self.d_k = d_k

        inv_freq = 1.0 / (
            theta
            ** (
                torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
                / d_k
            )
        )
        t = torch.arange(max_seq_len, device=device, dtype=torch.float32)

        freqs = torch.outer(t, inv_freq)

        emb = freqs.repeat_interleave(2, dim=-1)

        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def _rotate_every_two(self, x):
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)

    def forward(self, x, token_positions) :
        cos = self.cos_cached[token_positions]
        sin = self.sin_cached[token_positions]

        return x * cos + self._rotate_every_two(x) * sin



def softmax(v, i) -> torch.Tensor:
    v_max = torch.max(v, dim=i, keepdim=True).values
    v_safe = v - v_max

    exp_v = torch.exp(v_safe)

    sum_exp = torch.sum(exp_v, dim=i, keepdim=True)

    return exp_v / sum_exp


def Attention(Q,K,V,mask=None):

    QK = einsum(Q,K,"... n d_k , ... m d_k -> ... n m")

    d_k  = Q.shape[-1]
    QK = QK / math.sqrt(d_k)

    if mask is not None:
        QK = torch.where(mask, QK, -torch.inf)

    attn_weights = softmax(QK, i=-1)

    out = einsum(attn_weights, V, "... i j, ... j d_v -> ... i d_v")

    return out



class MultiHeadSelfAttention(nn.Module):
    def __init__(self,d_model:int,num_heads:int,device = None,dtype = None,theta=None, max_seq_len=None):
        super().__init__()

        self.d_model = d_model
        self.num_heads = num_heads

        self.d_k = d_model//num_heads

        factory_kwargs = {'device': device, 'dtype': dtype}
                
        self.W_Q = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.W_K = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.W_V = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))
        self.W_O = nn.Parameter(torch.empty((d_model, d_model), **factory_kwargs))

        if theta is None or max_seq_len is None:
            self.rope = None
        else:
            self.rope = RoPE(theta,self.d_k,max_seq_len)

        std = math.sqrt(1.0/d_model)
        
        torch.nn.init.trunc_normal_(
            self.W_Q,
            mean = 0.0,
            std = std,
            a = -3.0*std,
            b = 3.0*std
        )

        torch.nn.init.trunc_normal_(
            self.W_K,
            mean = 0.0,
            std = std,
            a = -3.0*std,
            b = 3.0*std
        )

        torch.nn.init.trunc_normal_(
            self.W_V,
            mean = 0.0,
            std = std,
            a = -3.0*std,
            b = 3.0*std
        )

        torch.nn.init.trunc_normal_(
            self.W_O,
            mean = 0.0,
            std = std,
            a = -3.0*std,
            b = 3.0*std
        )
        
        
    def forward(self,x,token_position = None):

        Q = einsum(self.W_Q,x,"d_out d_in , ... seq d_in -> ... seq d_out")
        K = einsum(self.W_K,x,"d_out d_in , ... seq d_in -> ... seq d_out")
        V = einsum(self.W_V,x,"d_out d_in , ... seq d_in -> ... seq d_out")


        Q = rearrange(Q," ... s (h d) -> ... h s d", h=self.num_heads)
        K = rearrange(K," ... s (h d) -> ... h s d", h=self.num_heads)
        V = rearrange(V," ... s (h d) -> ... h s d", h=self.num_heads)

        if token_position is None:
                token_position = torch.arange(Q.shape[-2])

        if self.rope is not None:
                    Q  = self.rope(Q,token_position)
                    K  = self.rope(K,token_position)

        seq = Q.shape[-2]


        mask = torch.tril(torch.ones(seq, seq, dtype=torch.bool,device = Q.device))

        score = Attention(Q,K,V,mask)

        score = rearrange(score,"... h s d -> ... s (h d)")

        output = einsum(self.W_O,score,"d_out d_in , ... seq d_in -> ... seq d_out")


        return output

class Transformer_Block(nn.Module):

    '''
    d_model: int,
    num_heads: int,
    d_ff: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
    in_features:
    '''
    def __init__(self,d_model,num_heads,d_ff,max_seq_len = None,theta = None):

        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads

        self.d_ff = d_ff

        self.ln1 = RMSNorm(self.d_model)
        self.ln2 = RMSNorm(self.d_model)

        self.theta = theta
        self.max_seq_len = max_seq_len

        self.attn = MultiHeadSelfAttention(self.d_model,self.num_heads,theta = self.theta,max_seq_len=self.max_seq_len)

        self.ffn = SwiGLU(self.d_model,self.d_ff)

    def forward(self,x):
        y=x+self.attn(self.ln1(x))

        y = y + self.ffn(self.ln2(y))

        return y


class Transformer_LM(nn.Module):
    '''
    vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        weights: dict[str, Tensor],
        in_indices: Int[Tensor, " batch_size sequence_length"],
    '''
    def __init__(self,vocab_size,context_length,num_layers,d_model,num_heads,d_ff,rope_theta):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size,d_model)

        self.ln_final = RMSNorm(d_model)

        self.lm_head = Linear(d_model,vocab_size)

        self.layers = nn.ModuleList(Transformer_Block(d_model,num_heads,d_ff,context_length,rope_theta) for i in range(num_layers))

    def forward(self,in_indices):
        x = self.token_embeddings(in_indices)

        for layer in self.layers:
            x = layer(x)

        x = self.ln_final(x)

        return self.lm_head(x)



