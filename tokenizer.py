import torch
from torch.utils.data import Dataset, DataLoader
import tiktoken
class GPTDatasetV1(Dataset):
   def __init__(self, txt, tokenizer, max_length, stride):
       self.input_ids = []
       self.target_ids = []
       token_ids = tokenizer.encode(txt)                         #A
       for i in range(0, len(token_ids) - max_length, stride):   #B
           input_chunk = token_ids[i:i + max_length]
           target_chunk = token_ids[i + 1: i + max_length + 1]
           self.input_ids.append(torch.tensor(input_chunk))
           self.target_ids.append(torch.tensor(target_chunk))
   def __len__(self):                                            #C
       return len(self.input_ids)
   def __getitem__(self, idx):                                   #D
       return self.input_ids[idx], self.target_ids[idx]
   

def create_dataloader_v1(txt, batch_size=4, max_length=256,
       stride=128, shuffle=True, drop_last=True, num_workers=0):
   tokenizer = tiktoken.get_encoding("gpt2")                   #A 
   dataset = GPTDatasetV1(txt, tokenizer, max_length, stride)    #B
   dataloader = DataLoader(
       dataset,
       batch_size=batch_size,
       shuffle=shuffle,
       drop_last=drop_last,                                      #C
       num_workers=0                                             #D
   )
   return dataloader



# with open("the-verdict.txt", "r", encoding="utf-8") as f:
#    raw_text = f.read()
#    vocab_size = 50257
# output_dim = 256
# token_embedding_layer = torch.nn.Embedding(vocab_size, output_dim)
# dataloader = create_dataloader_v1(
#    raw_text, batch_size=8, max_length=4, stride=4, shuffle=False)
# inputs, targets = next(iter(dataloader))
# print("Token IDs:\n", inputs)
# print("\nInputs shape:\n", inputs.shape)
# token_embeddings = token_embedding_layer(inputs)
# print(token_embeddings.shape)

# context_length = 4
# pos_embedding_layer = torch.nn.Embedding(context_length, output_dim)
# pos_embeddings = pos_embedding_layer(torch.arange(context_length))
# print(pos_embeddings.shape)

# input_embeddings = token_embeddings + pos_embeddings
# print(input_embeddings.shape)
