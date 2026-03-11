# Building a Small Language Model from Scratch: A Practical Guide to Domain-Specific AI

**By Abdul Sami** | Jan 11, 2025 | 4 min read

**Source:** https://medium.com/@rajasami408/building-a-small-language-model-from-scratch-a-practical-guide-to-domain-specific-ai-59539131437f

---

Right now, enterprise companies don't want to use OpenAI's ChatGPT model, Claude from Anthropic, or any other similar models. Even though these companies claim your data will be safe if you use their enterprise packages, there might still be concerns. Secondly, LLMs are very big in terms of size, making it hard for companies to create their own custom LLMs. So here, I am going to explain how you can build your own Small Language Model that will be trained only on your own data and won't require significant resources to run.

## Step 1: Gather and Prepare Data

First, you need to make a TXT file where you will gather all the data. For instance, the author scraped data from a company website because the goal was for the AI to provide answers only about that company.

After gathering all the data, train a tokenizer which will convert your words into tokens.

```python
# Initialize and train tokenizer
tokenizer = Tokenizer(BPE())
tokenizer.pre_tokenizer = Whitespace()
trainer = BpeTrainer(vocab_size=vocab_size-5)
special_tokens_dict = {
    "<|start|>": 0,
    "<|end|>": 1,
    "<|system|>": 2,
    "<|user|>": 3,
    "<|ai|>": 4
}
tokenizer.add_special_tokens(list(special_tokens_dict))
text_list = [text]
tokenizer.train_from_iterator(text_list, trainer)

print("Tokenizer trained")
print("Length of tokenizer:", len(tokenizer.get_vocab()))

def encode(s):
    return tokenizer.encode(s).ids

def decode(l):
    return tokenizer.decode(l)

# Create dataset
class TextDataset(Dataset):
    def __init__(self, data, context_length):
        self.data = torch.tensor(data, dtype=torch.long)
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length

    def __getitem__(self, idx):
        x = self.data[idx:idx + self.context_length]
        y = self.data[idx + 1:idx + self.context_length + 1]
        return x, y
```

## Step 2: Split Data into Training and Validation

```python
# Prepare data
encoded_text = encode(text)
train_size = int(len(encoded_text) * 0.9)
train_data = encoded_text[:train_size]
val_data = encoded_text[train_size:]

train_dataset = TextDataset(train_data, context_length)
val_dataset = TextDataset(val_data, context_length)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

print(f"Train dataset: {len(train_dataset)}")
print(f"Validation dataset: {len(val_dataset)}")
```

## Step 3: Attention Mechanism

Create an `Attention` class (called `Head`) with Key, Query, and Value components.

According to the embedding dimension, create Linear functions with appropriate dimensions. For example, if we use `nn.Linear(128, 12)`:

- Input size: 128 (embedding dimension)
- Output size: 12 (head size)

### Masking

Mask certain values in the attention mechanism to negative infinity (`-inf`) for causal/autoregressive attention — each token can only attend to itself and previous tokens. By setting future positions to `-inf`, they become `0` after softmax, ensuring the model can only look at past context.

```python
#  If attention matrix before masking is:
# [[1.0, 2.0, 3.0],
#  [4.0, 5.0, 6.0],
#  [7.0, 8.0, 9.0]]
# After applying causal mask (setting upper triangle to -inf):
# [[1.0,  -inf, -inf],
#  [4.0,  5.0,  -inf],
#  [7.0,  8.0,  9.0]]
# After softmax, the -inf values become 0:
# [[1.0, 0.0, 0.0],
#  [0.7, 0.3, 0.0],
#  [0.2, 0.3, 0.5]]
```

```python
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(embedding_dim, head_size)
        self.query = nn.Linear(embedding_dim, head_size)
        self.value = nn.Linear(embedding_dim, head_size)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)

        # Compute attention scores
        wei = q @ k.transpose(-2, -1) / np.sqrt(k.size(-1))
        mask = torch.triu(torch.ones(T, T), diagonal=1).bool().to(device)
        wei = wei.masked_fill(mask, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # Apply attention to values
        v = self.value(x)
        out = wei @ v
        return out
```

## Step 4: MultiHeadAttention, FeedForward, and TransformerBlock

You can increase the number of blocks to make the model bigger, but balance against your training data size to avoid overfitting (model too large) or underfitting (model too small).

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, head_size, num_heads):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, embedding_dim)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    def __init__(self, hidden_dim, num_heads):
        super().__init__()
        head_size = hidden_dim // num_heads
        self.attention = MultiHeadAttention(head_size, num_heads)
        self.ff = FeedForward(hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)

    def forward(self, x):
        x = x + self.attention(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x
```

## Step 5: The Small Language Model Class

Integrate all components into the final model class:

```python
class SmallLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.pos_embedding = nn.Embedding(context_length, embedding_dim)
        self.blocks = nn.Sequential(*[Block(embedding_dim, num_heads) for _ in range(num_layers)])
        self.ln_head = nn.Linear(embedding_dim, vocab_size)

    def forward(self, x, targets=None):
        B, T = x.shape

        # Get embeddings
        tok_emb = self.tok_embedding(x)
        pos_emb = self.pos_embedding(torch.arange(T, device=device))
        x = tok_emb + pos_emb

        # Apply transformer blocks
        x = self.blocks(x)
        logits = self.ln_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


# Initialize model and optimizer
model = SmallLanguageModel().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
```

## Step 6: Training Loop

```python
steps = 2000
for step in range(steps):
    x, y = get_batch('train')
    x, y = x.to(device), y.to(device)
    logits, loss = model(x, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if step % 100 == 0:
        print(f"Step {step}, Loss: {loss.item():.4f}")
    if step % 1000 == 0:
        print(f"Saving Model at Step {step}, Loss: {loss.item():.4f}")
        torch.save(model.state_dict(), f'drive/MyDrive/Small_Language_Model/pre_train_model_step_{step}.pth')
```

### Training Output

```
Step 0, Loss: 8.7914
Saving Model at Step 0, Loss: 8.7914
Step 100, Loss: 3.6767
Step 200, Loss: 2.4247
Step 300, Loss: 1.2641
Step 400, Loss: 0.7001
Step 500, Loss: 0.2650
Step 600, Loss: 0.1408
Step 700, Loss: 0.1231
Step 800, Loss: 0.1030
Step 900, Loss: 0.0929
Step 1000, Loss: 0.0837
Saving Model at Step 1000, Loss: 0.0837
Step 1100, Loss: 0.0762
Step 1200, Loss: 0.0710
Step 1300, Loss: 0.0661
Step 1400, Loss: 0.0619
Step 1500, Loss: 0.0574
Step 1600, Loss: 0.0564
Step 1700, Loss: 0.0473
Step 1800, Loss: 0.0520
Step 1900, Loss: 0.0472
```

## Step 7: Inference

The inference loop generates responses token by token using the trained model:

```python
model.eval()
query = "tell me about your company?"
input = encode(f"""<|start|><|system|>You are a conversational assistant at Cortech, prepared to discuss various tech services and innovations.<|user|>{query}<|ai|>""")
max_new_tokens = 200
current_token = 0
output = ""

while current_token < max_new_tokens:
    context = torch.tensor([input], dtype=torch.long, device=device)
    output_ids = model.generate(context, max_new_tokens=1)[0].tolist()
    if output_ids[-1] in [1, 2, 3, 4]:
        break
    current_token += 1
    input.append(output_ids[-1])
    output += decode([output_ids[-1]]) + " "

print("USER:", query)
print("AI:")
output
```

### Sample Outputs After Training

**Query: "tell me about your company?"**
```
USER: tell me about your company?
AI:
'Hello  Cortech provides a range of AI services including machine learning , deep learning ,
natural language processing , and generative AI . We use these technologies to build advanced
platforms that can enhance your business operations and customer interactions . Are you
interested in a specific type of AI application ?'
```

**Query: "My website isn't ranking well on Google."**
```
USER: My website isn't ranking well on Google.
AI:
'I see . This could be due to several factors . We can help by implementing our comprehensive
SEO strategy , which includes : 1 . Technical optimization of your website'
```

**Query: "I'm looking for some information about your Shopify services."**
```
USER: I'm looking for some information about your Shopify services.
AI:
'Great choice  We offer comprehensive Shopify migration services that include data transfer ,
theme customization , and payment gateway setup . Would you like to know more about our
migration process ?'
```

**Query: "What Services are you offering?"**
```
USER: What Services are you offering?
AI:
'We offer comprehensive AI services including machine learning , deep learning , natural language
processing , and cloud services . We specialize in developing custom AI solutions that can
transform business operations . Would you like to hear about any specific AI application ?'
```

**Query: "could you give me your contact details?"**
```
USER: could you give me your contact details?
AI:
'Of course  You can reach us at our Texas office at + 1 7373178366 or via email at
[email protected]  Our New York office can also be contacted at + 1 737312548 8 or
[email protected]  Which office would you prefer to contact ?'
```

## Summary

This project shows that even small language models, trained on limited domain-specific data, can deliver practical value for businesses. While there's room for improvement, the results demonstrate the potential of lightweight, privacy-preserving AI solutions.

## Key Takeaways

- Build a training corpus from domain-specific data (e.g., scraped company website)
- Train a BPE tokenizer with special tokens (`<|start|>`, `<|end|>`, `<|system|>`, `<|user|>`, `<|ai|>`)
- Implement causal self-attention (`Head`) with upper-triangular masking
- Compose `MultiHeadAttention`, `FeedForward`, and `Block` into a full transformer
- Assemble into `SmallLanguageModel` with token + positional embeddings
- Train with Adam optimizer; loss drops from ~8.8 to ~0.05 in 2,000 steps
- Generate responses autoregressively, stopping on special tokens
