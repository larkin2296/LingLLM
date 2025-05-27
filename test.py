import torch
from torch import nn

VOCAB_SIZE = 100
PAD_TOKEN_ID = 0
max_seq_len = 8

class MiniTransformerBlock(nn.Module):
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_size)
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        self.norm2 = nn.LayerNorm(embed_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + self.dropout(attn_out)
        x_norm2 = self.norm2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)
        return x

class MiniLLM(nn.Module):
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim)
        self.blocks = nn.ModuleList([
            MiniTransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids):
        batch, seq_len = input_ids.size()
        device = input_ids.device
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        logits = self.fc_out(x)
        return logits

device = torch.device("cpu")
model = MiniLLM(VOCAB_SIZE, embed_dim=16, max_seq_len=max_seq_len, num_heads=2, num_layers=2).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100)

# input: [10, 20, 30, 40, 50, 60, 70, 80]
# label: [-100, -100, -100, -100, -100, -100, 70, 80]，只学最后两个output
input_ids = torch.tensor([[10, 20, 30, 40, 50, 60, 70, 80]], dtype=torch.long).to(device)
labels = torch.tensor([[-100, -100, -100, -100, -100, -100, 70, 80]], dtype=torch.long).to(device)

for i in range(50):
    optimizer.zero_grad()
    logits = model(input_ids)
    loss = loss_fn(logits.view(-1, VOCAB_SIZE), labels.view(-1))
    print(f"epoch={i}, loss={loss.item():.4f}")
    loss.backward()
    optimizer.step()
