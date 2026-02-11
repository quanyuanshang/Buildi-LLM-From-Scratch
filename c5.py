from gpt_download import download_and_load_gpt2
from pretrain import GPT_CONFIG_124M
from gpt import GPTModel
import torch
import numpy as np
from tokenizer import text_to_token_ids, token_ids_to_text
from gpt import generate
import tiktoken
settings, params = download_and_load_gpt2(model_size="124M", models_dir="gpt2")
tokenizer = tiktoken.get_encoding("gpt2")
model_configs = {
   "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
   "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
   "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
   "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def assign(left, right):
   if left.shape != right.shape:
       raise ValueError(f"Shape mismatch. Left: {left.shape}, Right: {right.shape}")
   return torch.nn.Parameter(torch.tensor(right, dtype=left.dtype, device=left.device))



def load_weights_into_gpt(gpt, params):
   gpt.pos_emb.weight = assign(gpt.pos_emb.weight, params['wpe']) #A
   gpt.tok_emb.weight = assign(gpt.tok_emb.weight, params['wte'])
   
   for b in range(len(params["blocks"])):                        #B
       q_w, k_w, v_w = np.split(                                 #C
           (params["blocks"][b]["attn"]["c_attn"])["w"], 3, axis=-1)
       gpt.trf_blocks[b].att.W_query.weight = assign(
           gpt.trf_blocks[b].att.W_query.weight, q_w.T)
       gpt.trf_blocks[b].att.W_key.weight = assign(
           gpt.trf_blocks[b].att.W_key.weight, k_w.T)
       gpt.trf_blocks[b].att.W_value.weight = assign(
           gpt.trf_blocks[b].att.W_value.weight, v_w.T)
       q_b, k_b, v_b = np.split(
           (params["blocks"][b]["attn"]["c_attn"])["b"], 3, axis=-1)
       gpt.trf_blocks[b].att.W_query.bias = assign(
           gpt.trf_blocks[b].att.W_query.bias, q_b)
       gpt.trf_blocks[b].att.W_key.bias = assign(
           gpt.trf_blocks[b].att.W_key.bias, k_b)
       gpt.trf_blocks[b].att.W_value.bias = assign(
           gpt.trf_blocks[b].att.W_value.bias, v_b)
       gpt.trf_blocks[b].att.out_proj.weight = assign(
           gpt.trf_blocks[b].att.out_proj.weight, 
           params["blocks"][b]["attn"]["c_proj"]["w"].T)
       gpt.trf_blocks[b].att.out_proj.bias = assign(
           gpt.trf_blocks[b].att.out_proj.bias, 
           params["blocks"][b]["attn"]["c_proj"]["b"])
       gpt.trf_blocks[b].ff.layers[0].weight = assign(
           gpt.trf_blocks[b].ff.layers[0].weight,  params["blocks"][b]["mlp"]["c_fc"]["w"].T)
       gpt.trf_blocks[b].ff.layers[0].bias = assign(
           gpt.trf_blocks[b].ff.layers[0].bias, 
           params["blocks"][b]["mlp"]["c_fc"]["b"])
       gpt.trf_blocks[b].ff.layers[2].weight = assign(
           gpt.trf_blocks[b].ff.layers[2].weight, 
           params["blocks"][b]["mlp"]["c_proj"]["w"].T)
       gpt.trf_blocks[b].ff.layers[2].bias = assign(
           gpt.trf_blocks[b].ff.layers[2].bias, 
           params["blocks"][b]["mlp"]["c_proj"]["b"])
       gpt.trf_blocks[b].norm1.scale = assign(
           gpt.trf_blocks[b].norm1.scale, 
           params["blocks"][b]["ln_1"]["g"])
       gpt.trf_blocks[b].norm1.shift = assign(
           gpt.trf_blocks[b].norm1.shift, 
           params["blocks"][b]["ln_1"]["b"])
       gpt.trf_blocks[b].norm2.scale = assign(
           gpt.trf_blocks[b].norm2.scale, 
           params["blocks"][b]["ln_2"]["g"])
       gpt.trf_blocks[b].norm2.shift = assign(
           gpt.trf_blocks[b].norm2.shift, 
           params["blocks"][b]["ln_2"]["b"])
   gpt.final_norm.scale = assign(gpt.final_norm.scale, params["g"])
   gpt.final_norm.shift = assign(gpt.final_norm.shift, params["b"])
   gpt.out_head.weight = assign(gpt.out_head.weight, params["wte"]) #D


# model_name = "gpt2-small (124M)"
# NEW_CONFIG = GPT_CONFIG_124M.copy()
# NEW_CONFIG.update(model_configs[model_name])
# NEW_CONFIG.update({"context_length": 1024})
# NEW_CONFIG.update({"qkv_bias": True})
# gpt=GPTModel(NEW_CONFIG)
# gpt.eval()
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# load_weights_into_gpt(gpt, params)
# gpt.to(device)
# torch.manual_seed(123)
# idx = text_to_token_ids("Every effort moves you", tokenizer)
# idx = idx.to(device)

# # 使用更好的生成参数
# print("=" * 60)
# print("生成参数对比：")
# print("=" * 60)

# # 参数1：平衡的采样（推荐）
# print("\n1. 平衡采样 (temperature=0.7, top_k=50) - 推荐用于自然文本生成")
# print("-" * 60)
# token_ids = generate(
#    model=gpt,
#    idx=idx,
#    max_new_tokens=50,
#    context_size=NEW_CONFIG["context_length"],
#    top_k=50,
#    temperature=0.7
# )
# print(token_ids_to_text(token_ids, tokenizer))

# # 参数2：贪心采样（确定性）
# print("\n2. 贪心采样 (temperature=0.0, top_k=None) - 最确定的输出")
# print("-" * 60)
# idx = text_to_token_ids("Every effort moves you", tokenizer)
# idx = idx.to(device)
# token_ids = generate(
#    model=gpt,
#    idx=idx,
#    max_new_tokens=50,
#    context_size=NEW_CONFIG["context_length"],
#    top_k=None,
#    temperature=0.0
# )
# print(token_ids_to_text(token_ids, tokenizer))

# # 参数3：更多随机性
# print("\n3. 更多随机性 (temperature=1.0, top_k=40) - 更多样化的输出")
# print("-" * 60)
# idx = text_to_token_ids("Every effort moves you", tokenizer)
# idx = idx.to(device)
# token_ids = generate(
#    model=gpt,
#    idx=idx,
#    max_new_tokens=50,
#    context_size=NEW_CONFIG["context_length"],
#    top_k=40,
#    temperature=1.0
# )
# print(token_ids_to_text(token_ids, tokenizer))

# print("\n" + "=" * 60)
# print("参数说明：")
# print("=" * 60)
# print("- temperature: 控制随机性 (0=确定, 1.0=正常, >1=更随机)")
# print("- top_k: 只从概率最高的k个token中采样 (None=全部)")
# print("- 推荐：temperature=0.7-0.9, top_k=40-50 用于高质量文本生成")
