import torch
print(torch.version.cuda)
from torch import nn
import torch.nn.functional as F


batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
if torch.cuda.is_available():
    print("we have a go for cuda")
else:
    print('THATS A NEGATIVE HOUSTON')

eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2
# ------------

with open('input.txt', 'r', encoding='utf-8') as file:
    text = file.read()

# here are all the unique characters that occur in this text

char = sorted(list(set(text)))
vocab_size = len(char)
stoi = {s: i for i, s in enumerate(char)}
itos = {i: s for s, i in stoi.items()}

encode = lambda x: [stoi[i] for i in x]
decode = lambda x: ''.join([itos[i] for i in x])

# Train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading
def get_batch(split):
    data = train_data if split == 'train' else val_data
    xi = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in xi])
    y = torch.stack([data[i+1:i+1+block_size] for i in xi])
    x, y = x.to(device), y.to(device)
    return x, y
@torch.no_grad()
def estimate_loss():

    out = {}
    m.eval()
    for split in ['train', 'eval']:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            logits, loss = m(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size,block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self,x):
        B,T,C = x.shape
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)

        wei = q @ k.transpose(-2,-1) * C**-0.5 # (B,T,C) @ (B,C,T) = (B,T,T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        out = wei @ v # (B,T,T) @ (B,T,C) = (B,T,C)
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
        nn.Linear(n_embd, 4 * n_embd),
        nn.ReLU(),
        nn.Linear(4 * n_embd, n_embd),
        nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.net(x)
        return x


class MultiHeadAttention(nn.Module):
    def __init__(self, num_head, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_head)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out



class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, n_embd)
        self.pos_embedding = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head) for _ in range(n_layer)])
        self.ln = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, target=None):
        B,T = idx.shape
        tok_emb = self.embedding(idx)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln(x)
        logits = self.lm_head(x)

        if target is None:
            loss = None
        else:
            B,T,C = logits.shape
            logits = logits.view(B*T,C)

            target = target.view(B*T)
            loss = F.cross_entropy(logits, target)

        return logits, loss

    def generate(self, idx, max_new_tokens):
            #idx is (B,T)
        for i in range(max_new_tokens):
            idx = idx[:,-block_size:] # since idx is (B,T)
            logits, loss = self(idx)

            logits = logits[:, -1, :]
            proba = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(proba, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


model = BigramLanguageModel()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=3e-4)

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['eval']:.4f}")


    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)

    loss.backward()
    optimizer.step()

# context = torch.zeros(1,1)
# decode(m.generate(context, max_new_tokens=500).tolist())





