import torch
import tiktoken
from c5 import GPTModel, load_weights_into_gpt
from gpt_download import download_and_load_gpt2

# ============ 方法1：加载微调后的模型（推荐用于分类任务）============

def load_finetuned_model(model_path, device):
    """
    加载微调后的分类模型
    
    Args:
        model_path: 模型权重文件路径 (e.g., "review_classifier.pth")
        device: 设备 (cuda 或 cpu)
    
    Returns:
        model: 加载好的模型
    """
    # 模型配置
    BASE_CONFIG = {
        "vocab_size": 50257,
        "context_length": 1024,
        "drop_rate": 0.0,
        "qkv_bias": True,
        "emb_dim": 768,
        "n_layers": 12,
        "n_heads": 12,
    }
    
    # 创建模型
    model = GPTModel(BASE_CONFIG)
    
    # 加载预训练权重
    model_size = "124M"
    settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
    load_weights_into_gpt(model, params)
    
    # 冻结预训练层
    for param in model.parameters():
        param.requires_grad = False
    
    # 添加分类头
    num_classes = 2
    model.out_head = torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"], out_features=num_classes)
    
    # 加载微调后的权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    
    # 移到设备
    model.to(device)
    model.eval()
    
    return model


# ============ 方法2：使用加载的模型进行预测 ============

def classify_text(text, model, tokenizer, device, max_length):
    """
    使用加载的模型进行文本分类
    
    Args:
        text: 输入文本
        model: 加载的模型
        tokenizer: tokenizer
        device: 设备
        max_length: 最大长度
    
    Returns:
        prediction: "spam" 或 "not spam"
    """
    model.eval()
    
    # 编码文本
    input_ids = tokenizer.encode(text)
    
    # 截断和填充
    input_ids = input_ids[:max_length]
    input_ids += [50256] * (max_length - len(input_ids))
    
    # 转换为张量
    input_tensor = torch.tensor(input_ids, device=device, dtype=torch.long).unsqueeze(0)
    
    # 推理
    with torch.no_grad():
        logits = model(input_tensor)[:, -1, :]
        logits = model.out_head(logits)
        predicted_label = torch.argmax(logits, dim=-1).item()
    
    return "spam" if predicted_label == 1 else "not spam"


# ============ 使用示例 ============

if __name__ == "__main__":
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型
    print("Loading model...")
    model = load_finetuned_model("review_classifier.pth", device)
    print("Model loaded successfully!")
    
    # 加载 tokenizer
    tokenizer = tiktoken.get_encoding("gpt2")
    
    # 设置最大长度（需要与训练时一致）
    max_length = 128  # 根据你的训练数据调整
    
    # 测试预测
    test_texts = [
        "You are a winner you have been specially selected to receive $1000 cash or a $2000 award.",
        "Hi, how are you doing today?",
        "Congratulations! You've won a free iPhone. Click here to claim your prize.",
    ]
    
    print("\n" + "="*60)
    print("Classification Results:")
    print("="*60)
    
    for text in test_texts:
        prediction = classify_text(text, model, tokenizer, device, max_length)
        print(f"\nText: {text[:50]}...")
        print(f"Prediction: {prediction}")
