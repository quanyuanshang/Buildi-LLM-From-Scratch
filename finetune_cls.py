import pandas as pd
import torch
from torch.utils.data import Dataset
import tiktoken
from gpt import generate
from tokenizer import text_to_token_ids, token_ids_to_text

data_file_path = "sms_spam_collection/SMSSpamCollection.tsv"   
df = pd.read_csv(data_file_path, sep="\t", header=None, names=["Label", "Text"])
def create_balanced_dataset(df):
    num_spam = df[df["Label"] == "spam"].shape[0] #A
    ham_subset = df[df["Label"] == "ham"].sample(num_spam, random_state=123) #B
    balanced_df = pd.concat([ham_subset, df[df["Label"] == "spam"]]) #C
    return balanced_df
balanced_df = create_balanced_dataset(df)
# print(balanced_df["Label"].value_counts())


balanced_df["Label"] = balanced_df["Label"].map({"ham": 0, "spam": 1})
def random_split(df, train_frac, validation_frac):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True) #A
    train_end = int(len(df) * train_frac) #B
    validation_end = train_end + int(len(df) * validation_frac)#C
    train_df = df[:train_end]
    validation_df = df[train_end:validation_end]
    test_df = df[validation_end:]
    return train_df, validation_df, test_df
train_df, validation_df, test_df = random_split(balanced_df, 0.7, 0.1) #D

train_df.to_csv("train.csv", index=None)
validation_df.to_csv("validation.csv", index=None)
test_df.to_csv("test.csv", index=None)


class SpamDataset(Dataset):
    def __init__(self, csv_file, tokenizer, max_length=None, pad_token_id=50256):
        self.data = pd.read_csv(csv_file)
        self.encoded_texts = [tokenizer.encode(text) for text in self.data["Text"]]
        
        # 确定最大长度
        if max_length is None:
            self.max_length = self._longest_encoded_length()
        else:
            self.max_length = max_length
            
        self.pad_token_id = pad_token_id

    def _longest_encoded_length(self):
        return max(len(encoded_text) for encoded_text in self.encoded_texts)

    def __getitem__(self, index):
        encoded = self.encoded_texts[index]
        
        # 1. 截断
        encoded = encoded[:self.max_length]
        
        # 2. 记录真实长度（重要！）
        actual_length = len(encoded)
        
        # 3. 填充
        padded_encoded = encoded + [self.pad_token_id] * (self.max_length - len(encoded))
        
        label = self.data.iloc[index]["Label"]
        
        # 返回三个值：输入ID，标签，真实长度
        return (
            torch.tensor(padded_encoded, dtype=torch.long),
            torch.tensor(label, dtype=torch.long),
            torch.tensor(actual_length, dtype=torch.long) 
        )

    def __len__(self):
        return len(self.data)
    
tokenizer = tiktoken.get_encoding("gpt2")
train_dataset = SpamDataset(csv_file="train.csv",max_length=None,tokenizer=tokenizer)
val_dataset = SpamDataset(csv_file="validation.csv",max_length=train_dataset.max_length,tokenizer=tokenizer)
test_dataset = SpamDataset(csv_file="test.csv",max_length=train_dataset.max_length,tokenizer=tokenizer)
from torch.utils.data import DataLoader
num_workers = 0
batch_size = 8
torch.manual_seed(123)
train_loader = DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers,drop_last=True,)
val_loader = DataLoader(dataset=val_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,)
test_loader = DataLoader(dataset=test_dataset,batch_size=batch_size,num_workers=num_workers,drop_last=False,)

# print("Input batch dimensions:", input_batch.shape)
# print("Label batch dimensions", target_batch.shape)

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

# 确保数据集的最大长度不超过模型的上下文长度
assert train_dataset.max_length <= BASE_CONFIG["context_length"], (
    f"Dataset length {train_dataset.max_length} exceeds model's context "
    f"length {BASE_CONFIG['context_length']}. Reinitialize data sets with "
    f"max_length={BASE_CONFIG['context_length']}"
)

from gpt_download import download_and_load_gpt2
from c5 import GPTModel, load_weights_into_gpt
model_size = CHOOSE_MODEL.split(" ")[-1].lstrip("(").rstrip(")")
settings, params = download_and_load_gpt2(model_size=model_size, models_dir="gpt2")
model = GPTModel(BASE_CONFIG)
load_weights_into_gpt(model, params)

for param in model.parameters():
    param.requires_grad = False

torch.manual_seed(123)
num_classes = 2
model.out_head= torch.nn.Linear(in_features=BASE_CONFIG["emb_dim"],out_features=num_classes)

for param in model.trf_blocks[-1].parameters():
    param.requires_grad = True
for param in model.final_norm.parameters():
    param.requires_grad = True


def calc_loss_batch(input_batch, target_batch, lengths, model, device):
    input_batch, target_batch = input_batch.to(device), target_batch.to(device)
    lengths = lengths.to(device) # lengths 也在 device 上
    
    logits = model(input_batch) # Shape: [batch_size, seq_len, vocab_size]
    
    # 获取每个样本最后一个真实 token 的索引
    # 例如：长度是 5，索引就是 4
    batch_indices = torch.arange(input_batch.shape[0], device=device)
    last_token_indices = lengths - 1
    
    # 选出对应的 logits
    # 结果 shape: [batch_size, vocab_size]
    selected_logits = logits[batch_indices, last_token_indices, :]
    
    loss = torch.nn.functional.cross_entropy(selected_logits, target_batch)
    return loss


def calc_loss_loader(data_loader, model, device, num_batches=None):
    total_loss = 0.
    if len(data_loader) == 0:
        return float("nan")
    elif num_batches is None:
        num_batches = len(data_loader)                         # A
    else:
        num_batches = min(num_batches, len(data_loader))

    for i, (input_batch, target_batch, lengths) in enumerate(data_loader):
        if i < num_batches:
            loss = calc_loss_batch(input_batch, target_batch, lengths, model, device)
            total_loss += loss.item()
        else:
            break

    return total_loss / num_batches

def calc_accuracy_loader(data_loader, model, device, num_batches=None):
    model.eval()
    correct_predictions, num_examples = 0, 0
    if num_batches is None: num_batches = len(data_loader)
    else: num_batches = min(num_batches, len(data_loader))
        
    # 注意这里解包增加了 lengths
    for i, (input_batch, target_batch, lengths) in enumerate(data_loader):
        if i < num_batches:
            input_batch, target_batch = input_batch.to(device), target_batch.to(device)
            lengths = lengths.to(device)
            
            with torch.no_grad():
                logits = model(input_batch)
                # 同样的逻辑：选择最后一个真实 Token
                batch_indices = torch.arange(input_batch.shape[0], device=device)
                selected_logits = logits[batch_indices, lengths - 1, :]
                
                predicted_labels = torch.argmax(selected_logits, dim=-1)
                num_examples += predicted_labels.shape[0]
                correct_predictions += (predicted_labels == target_batch).sum().item()
        else:
            break
    return correct_predictions / num_examples

def train_classifier_simple(model, train_loader, val_loader, optimizer, device,
                            num_epochs, eval_freq, eval_iter, tokenizer):
    # 初始化记录列表
    train_losses, val_losses, train_accs, val_accs = [], [], [], []
    examples_seen, global_step = 0, 0

    # 主训练循环
    for epoch in range(num_epochs):                          # A
        model.train()

        for input_batch, target_batch, lengths in train_loader:       # B
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, lengths, model, device)
            loss.backward()                                  # C
            optimizer.step()                                 # D
            examples_seen += input_batch.shape[0]            # E
            global_step += 1

            if global_step % eval_freq == 0:                 # F
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter)
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                print(f"Ep {epoch+1} Step {global_step:06d}: "
                      f"Train loss {train_loss:.3f}, val loss {val_loss:.3f}")

        # 每个 epoch 结束后计算准确率
        train_accuracy = calc_accuracy_loader(               # G
            train_loader, model, device, num_batches=eval_iter)

        val_accuracy = calc_accuracy_loader(
            val_loader, model, device, num_batches=eval_iter)

        print(f"Training accuracy: {train_accuracy*100:.2f}% | ", end="")
        print(f"Validation accuracy: {val_accuracy*100:.2f}%")

        train_accs.append(train_accuracy)
        val_accs.append(val_accuracy)

    return train_losses, val_losses, train_accs, val_accs, examples_seen


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
# torch.manual_seed(123)
# train_accuracy = calc_accuracy_loader(train_loader, model, device, num_batches=10)
# val_accuracy = calc_accuracy_loader(val_loader, model, device, num_batches=10)
# test_accuracy = calc_accuracy_loader(test_loader, model, device, num_batches=10)
# print(f"Training accuracy: {train_accuracy*100:.2f}%")
# print(f"Validation accuracy: {val_accuracy*100:.2f}%")
# print(f"Test accuracy: {test_accuracy*100:.2f}%")


def evaluate_model(model, train_loader, val_loader, device, eval_iter):
    model.eval()
    with torch.no_grad():
        train_loss = calc_loss_loader(train_loader, model, device,num_batches=eval_iter)
        val_loss = calc_loss_loader(val_loader, model, device, num_batches=eval_iter)
        model.train()
    return train_loss, val_loss

import time
import torch

# 记录开始时间
start_time = time.time()

# 设置随机种子保证可复现
torch.manual_seed(123)

# 定义优化器（AdamW）
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=5e-5,
    weight_decay=0.1
)

# 训练轮数
num_epochs = 5

# 调用训练函数
train_losses, val_losses, train_accs, val_accs, examples_seen = train_classifier_simple(
    model, train_loader, val_loader, optimizer, device,
    num_epochs=num_epochs, eval_freq=50, eval_iter=5,
    tokenizer=tokenizer
)

# 记录结束时间并计算耗时
end_time = time.time()
execution_time_minutes = (end_time - start_time) / 60
print(f"Training completed in {execution_time_minutes:.2f} minutes.")

torch.save(model.state_dict(), "review_classifier.pth")


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
text_1 = ("You are a winner you have been specially"" selected to receive $1000 cash or a $2000 award.")
print(classify_review(text_1, model, tokenizer, device, max_length=train_dataset.max_length))