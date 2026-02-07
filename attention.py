import torch
inputs = torch.tensor(
[[0.43, 0.15, 0.89], # Your    (x^1)
[0.55, 0.87, 0.66], # journey  (x^2)
[0.57, 0.85, 0.64], # starts   (x^3)
[0.22, 0.58, 0.33], # with     
[0.77, 0.25, 0.10], # one      
[0.05, 0.80, 0.55]] # step     
)
# part1
query = inputs[1]                                               
attn_scores = torch.empty(6, 6)
attn_scores = inputs @ inputs.T
attn_weights = torch.softmax(attn_scores, dim=-1)# dim=-1, we are instructing the softmax function to apply the normalization along the last dimension of the attn_scores tensor. 
all_context_vecs = attn_weights @ inputs
# print(all_context_vecs)

#part2
x_2 = inputs[1]                                                   
d_in = inputs.shape[1]                                            
d_out = 2
torch.manual_seed(123)
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_key   = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)                  
query_2 = x_2 @ W_query 
keys = inputs @ W_key 
values = inputs @ W_value
attn_scores_2 = query_2 @ keys.T 
d_k = keys.shape[-1]
attn_weights_2 = torch.softmax(attn_scores_2 / d_k**0.5, dim=-1)
context_vec_2 = attn_weights_2 @ values

# a compact class
import torch.nn as nn
class SelfAttention_v1(nn.Module):
   def __init__(self, d_in, d_out, qkv_bias=False):
       super().__init__()
       self.d_out = d_out
       self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)#  nn.Linear has an optimized weight initialization scheme, contributing to more stable and effective model training
       self.W_key   = nn.Linear(d_in, d_out, bias=qkv_bias)
       self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
   def forward(self, x):
       keys = x @ self.W_key
       queries = x @ self.W_query
       values = x @ self.W_value
       attn_scores = queries @ keys.T # omega
       attn_weights = torch.softmax(
           attn_scores / keys.shape[-1]**0.5, dim=-1)
       context_vec = attn_weights @ values
       return context_vec
   
# part3 causal class
# batch = torch.stack((inputs, inputs), dim=0)
# print(batch.shape)

class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout) #A
        self.register_buffer('mask',torch.triu(torch.ones(context_length, context_length),diagonal=1)) #B
    def forward(self, x):
        b, num_tokens, d_in = x.shape #C
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries @ keys.transpose(1, 2) #C
        attn_scores.masked_fill_( #D
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        return context_vec
    

#part4 multi-head class
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length,dropout, num_heads, qkv_bias=False):
        super().__init__()
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias)
                                    for _ in range(num_heads)])
        
    def forward(self, x):
        return torch.cat([head(x) for head in self.heads], dim=-1)
    
#part5 An efficient multi-head attention class
class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length,dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask',torch.triu(torch.ones(context_length, context_length),diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        queries = self.W_query(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.W_value(x).view(b, num_tokens, self.num_heads, self.head_dim).transpose(1, 2)
        attn_scores = queries @ keys.transpose(-2, -1)
        attn_scores.masked_fill_(
            self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)
        attn_weights = torch.softmax(attn_scores / self.head_dim**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)
        context_vec = attn_weights @ values
        context_vec = context_vec.transpose(1, 2).contiguous().view(b, num_tokens, self.d_out)
        return context_vec
    
# a = torch.tensor([[[[0.2745, 0.6584, 0.2775, 0.8573],[0.8993, 0.0390, 0.9268, 0.7388],[0.7179, 0.7058, 0.9156, 0.4340]],[[0.0772, 0.3565, 0.1479, 0.5331],[0.4066, 0.2318, 0.4545, 0.9737],[0.4606, 0.5159, 0.4220, 0.5786]]]])
# first_head = a[0, 0, :, :]
# first_res = first_head @ first_head.T
# second_head = a[0, 1, :, :]
# second_res = second_head @ second_head.T
# print("\nSecond head:\n", second_res)