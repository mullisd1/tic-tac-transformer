import torch
from torch import nn
import torch.nn.functional as F


class Head(nn.Module):
    """ Singular Head"""

    def __init__(self,
                 num_embedding,
                 head_size,
                 dropout
                 ):
        super().__init__()
        self.key = nn.Linear(num_embedding, head_size, bias=False)
        self.query = nn.Linear(num_embedding, head_size, bias=False)
        self.value = nn.Linear(num_embedding, head_size, bias=False)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B,T,C = x.shape

        k = self.key(x) # (B,T,C)
        q = self.query(x) # (B,T,C)

        # Computer attention
        # C**0.5 is to make the numbers smaller so softmax doesn't do weird things
        wei = q @ k.transpose(-2, -1) / C**0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)

        # 
        V = self.value(x)
        out = wei @ V # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out
    
class MultiHeadAttention(nn.Module):
    """Multiple Attention Heads"""
    def __init__(self, num_heads, num_embedding, head_size, dropout):
        super().__init__()
        self.heads = nn.ModuleList([Head(num_embedding, head_size, dropout) for i in range(num_heads)])
        self.project = nn.Linear(num_embedding, num_embedding)                  # Projection layer for gettting back into the residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.project(out)
        out = self.dropout(out)
        return out
    
class FeedFoward(nn.Module):
    """Single Layer"""

    def __init__(self, num_embedding, dropout):
        super().__init__()

        self.m = nn.Sequential(nn.Linear(num_embedding, 4 * num_embedding), # 4* because they did it in the paper
                               nn.ReLU(),
                               nn.Linear(4 * num_embedding, num_embedding), # Projection layer for gettting back into the residual pathway
                               nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.m(x)
    
class Block(nn.Module):
    """"""

    def __init__(self, num_embedding, num_heads, dropout):
        super().__init__()

        head_size = num_embedding // num_heads
        self.self_attn = MultiHeadAttention(num_heads, num_embedding, head_size, dropout)
        self.feed_fwd = FeedFoward(num_embedding, dropout)

        self.lay_norm1 = nn.LayerNorm(num_embedding)
        self.lay_norm2 = nn.LayerNorm(num_embedding)

    def forward(self, x):
        x = x + self.self_attn(self.lay_norm1(x))
        x = x + self.feed_fwd(self.lay_norm2(x))
        return x
    
class TicTacToeModel(nn.Module):
    def __init__(self,
                 vocab_size,
                 block_size,
                 num_embedding,
                 num_heads,
                 num_blocks,
                 dropout,
                ):
        super().__init__()

        # Embeddings
        self.token_embedding_table = nn.Embedding(vocab_size,num_embedding)
        self.position_embedding_table = nn.Embedding(block_size, num_embedding)

        # Attention
        self.attn_blocks = nn.Sequential(
            *[Block(num_embedding, num_heads, dropout) for i in range(num_blocks)],
            nn.LayerNorm(num_embedding),
        )

        self.lm_head = nn.Linear(num_embedding, vocab_size)

    def forward(self, inputs):
        B, T = inputs.shape

        token_embedding = self.token_embedding_table(inputs) # (B, T, C)
        position_embedding = self.position_embedding_table(torch.arange(T)) # (T, C)
        x = token_embedding + position_embedding # (B, T, C)

        x = self.attn_blocks(x)
        logits = self.lm_head(x) # (B, T, vocab_size)
            
        return logits
