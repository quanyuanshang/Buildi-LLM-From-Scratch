from importlib.metadata import version
import os
import tiktoken

HOST = "openaipublic.blob.core.windows.net"

def ensure_no_proxy(host: str) -> None:
	for key in ("NO_PROXY", "no_proxy"):
		current = os.environ.get(key, "")
		items = [h.strip() for h in current.split(",") if h.strip()]
		if host not in items:
			os.environ[key] = ",".join(items + [host]) if items else host

for key in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
	if os.environ.get(key):
		print(f"Proxy env {key} is set; bypassing proxy for {HOST}.")

ensure_no_proxy(HOST)

# print("tiktoken version:", version("tiktoken"))
tokenizer = tiktoken.get_encoding("gpt2")
# text = "someunknownPlace."
# integers = tokenizer.encode(text, allowed_special={"<|endoftext|>"})
# print(integers)
# print(tokenizer.decode(integers))
with open("the-verdict.txt", "r", encoding="utf-8") as f:
    raw_text = f.read()

enc_text = tokenizer.encode(raw_text)
print(len(enc_text))
enc_sample = enc_text[50:]
context_size = 4 #A
x = enc_sample[:context_size]
y = enc_sample[1:context_size+1]
print(f"x: {x}")
print(f"y:      {y}")
for i in range(1, context_size+1):
    context = enc_sample[:i]
    desired = enc_sample[i]
    print(context, "---->", desired)