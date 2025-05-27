import torch
import torch.nn as nn

class MiniTransformerBlock(nn.Module):
    """
    最小的Transformer Block：包含自注意力、前馈网络、残差连接(防止梯度消散,提高模型效率)和LayerNorm(用来实现对 tensor张量 的层标准化)
    """
    def __init__(self, embed_size, num_heads, dropout=0.1):
        super().__init__()
        # 多头自注意力机制
        # embed_dim: 模型的总维度, num_heads: 多头注意力的头数, batch_first – 如果为 True，则输入和输出 tensor 以 (batch, seq, feature) 形式提供。默认值：False（seq, batch, feature）。
        self.attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=num_heads, batch_first=True)
        # 第一层LayerNorm
        self.norm1 = nn.LayerNorm(embed_size)
        # 前馈网络，两层全连接加激活
        self.ff = nn.Sequential(
            nn.Linear(embed_size, embed_size * 4),
            nn.GELU(),
            nn.Linear(embed_size * 4, embed_size)
        )
        # 第二层LayerNorm
        self.norm2 = nn.LayerNorm(embed_size)
        # 两次LayerNorm
        # 每个子层都单独正则化，防止训练不稳定/特征爆炸
        # 保证每个“子模块输出”都有机会归一化，减少梯度消失/爆炸
        # 残差+归一化会让模型能更深、效果更好
        # 是现代Transformer（包括BERT、GPT等）训练好效果的“秘诀”之一
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None, key_padding_mask=None):
        # x形状：[batch, seq_len, embed_dim]
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm,attn_mask=attn_mask, key_padding_mask=key_padding_mask)    # padding mask (标记pad位置))
        # print(f"attn_weights: shape={attn_weights.shape} min={attn_weights.min().item():.4f} max={attn_weights.max().item():.4f} mean={attn_weights.mean().item():.4f}")
        # zero_count = (attn_weights == 0).sum().item()
        # nan_count = torch.isnan(attn_weights).sum().item()
        # inf_count = torch.isinf(attn_weights).sum().item()
        # all_zero_rows = ((attn_weights == 0).sum(dim=-1) == attn_weights.size(-1)).sum().item()
        # print(f"zero={zero_count} nan={nan_count} inf={inf_count} all_zero_rows={all_zero_rows}")
        x = x + self.dropout(attn_out)
        # ...同理FFN...
        x_norm2 = self.norm2(x)
        ff_out = self.ff(x_norm2)
        x = x + self.dropout(ff_out)
        return x
    
class MiniLLM(nn.Module):
    """
    最小语言模型：分词嵌入(词汇映射到实数向量的方法总称)、位置编码、一个Transformer Block和输出层
    """
    def __init__(self, vocab_size, embed_dim, max_seq_len, num_heads, num_layers=2, dropout=0.1):
        super().__init__()
        self.max_seq_len = max_seq_len
        # vocab_size字典大小
        self.embedding = nn.Embedding(vocab_size, embed_dim) # 词嵌入
        # max_seq_len输入大小
        self.pos_embedding = nn.Embedding(max_seq_len, embed_dim) # 位置编码
        self.blocks = nn.ModuleList([
            MiniTransformerBlock(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_layers)
        ])
        # self.block = MiniTransformerBlock(embed_dim, num_heads) # Transformer Block
        self.fc_out = nn.Linear(embed_dim, vocab_size)

    def forward(self, input_ids, attention_mask=None):
        batch, seq_len = input_ids.size()
        device = input_ids.device

        # start (Number, 可选) – 点集的起始值。默认值：0。
        # end (Number) – 点集的结束值
        # step (Number, 可选) – 每对相邻点之间的间隔。默认值：1。
        # 返回一个一维张量，包含从 start 到 end 的数字，步长为 step。
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch, -1)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        # print("embedding输出max:", x.max().item())
        # print("embedding输出min:", x.min().item())
        # print("embedding输出nan数:", torch.isnan(x).sum().item())
        # print("embedding输出inf数:", torch.isinf(x).sum().item())

        if torch.isnan(x).any() or torch.isinf(x).any():
            print("embedding输出异常，检查embedding权重、token id合法性！")
        
        # causal mask：[seq, seq]
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        causal_mask = causal_mask.masked_fill(causal_mask == 1, float('-inf'))
        
        # attention_mask: [batch, seq]，padding位置是0
        key_padding_mask = torch.where(attention_mask == 0, torch.full_like(attention_mask, float('-inf'), dtype=torch.float), torch.zeros_like(attention_mask, dtype=torch.float))
        for block in self.blocks:
            x = block(x, attn_mask=causal_mask, key_padding_mask=key_padding_mask)
        logits = self.fc_out(x)
        return logits
