import torch
import torch.nn as nn
import tiktoken
from c5 import GPTModel

CHOOSE_MODEL = "gpt2-small (124M)"
INPUT_PROMPT = "Every effort moves"

BASE_CONFIG = {
    "vocab_size": 50257,        # 词表大小
    "context_length": 1024,     # 上下文长度
    "drop_rate": 0.0,           # Dropout 概率
    "qkv_bias": True            # QKV 偏置
}

model_configs = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

# 更新基础配置为所选模型的参数
BASE_CONFIG.update(model_configs[CHOOSE_MODEL])

model=GPTModel(BASE_CONFIG)
num_classes = 2
model.out_head= torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"],out_features=num_classes)
model.load_state_dict(torch.load("review_classifier.pth"))
tokenizer = tiktoken.get_encoding("gpt2")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
def classify_review(text, model, tokenizer, device, max_length=None, pad_token_id=50256):
    model.eval()
    
    input_ids = tokenizer.encode(text)
    supported_context_length = model.pos_emb.weight.shape[1]
    
    # 截断
    limit = min(max_length, supported_context_length) if max_length else supported_context_length
    input_ids = input_ids[:limit]
    
    # 记录真实长度（未填充前）
    actual_length = len(input_ids)

    # 填充
    input_ids += [pad_token_id] * (limit - len(input_ids))
    
    input_tensor = torch.tensor(input_ids, device=device).unsqueeze(0)
    
    with torch.no_grad():
        logits = model(input_tensor)
        # 关键：取 actual_length - 1 的位置，而不是 -1
        last_token_logits = logits[:, actual_length - 1, :]
        predicted_label = torch.argmax(last_token_logits, dim=-1).item()
        
    return "spam" if predicted_label == 1 else "ham"
text_1 = ("Ew are you one of them?")
print(classify_review(text_1, model, tokenizer, device, max_length=120))
