import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention with variable-length groups defined by batch indices.
    Similar to torch.nn.MultiheadAttention but operates on sets/graphs.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Shared projections: project to Q, K, V in one go
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)        
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_indices):
        """
        Args:
            x: [N, embed_dim] node embeddings
            batch_indices: [N] tensor with graph/set ids for each node
        Returns:
            out: [N, embed_dim] updated node embeddings
        """
        N, _ = x.size()

        # Project to Q, K, V and split
        qkv = self.qkv_proj(x)  # [N, 3*embed_dim]
        q, k, v = qkv.chunk(3, dim=-1)  # each [N, embed_dim]

        # Reshape for multi-head
        q = q.view(N, self.num_heads, self.head_dim)
        k = k.view(N, self.num_heads, self.head_dim)
        v = v.view(N, self.num_heads, self.head_dim)

        out = torch.zeros_like(q)

        # Process each group separately
        for b in torch.unique(batch_indices):
            mask = batch_indices == b
            q_b, k_b, v_b = (
                q[mask],
                k[mask],
                v[mask],
            )  # [num_nodes_b, num_heads, head_dim]

            # Compute scaled dot-product attention
            attn_scores = torch.einsum("nhd,mhd->hnm", q_b, k_b) / (self.head_dim**0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Aggregate values
            out_b = torch.einsum(
                "hnm,mhd->nhd", attn_probs, v_b
            )  # [num_nodes_b, num_heads, head_dim]
            out[mask] = out_b

        # Merge heads back
        out = out.reshape(N, self.embed_dim)
        out = self.out_proj(out)
        return out


class AttentionPoolingBlock(nn.Module):
    """
    Multi-head attention pooling per group using batch indices.
    Each group is pooled into one or more seed embeddings.
    """

    def __init__(self, embed_dim, num_heads, num_seeds=1, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.num_seeds = num_seeds

        # Shared linear projections for K and V
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

        # Learnable seed vectors
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, embed_dim))
        # Linear to project seeds to multi-head Q
        self.q_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x, batch_indices):
        """
        Args:
            x: [N, embed_dim] node embeddings
            batch_indices: [N] group IDs
        Returns:
            pooled: [num_groups * num_seeds, embed_dim]
        """
        N, _ = x.size()

        # Project K and V once
        K = self.k_proj(x).view(N, self.num_heads, self.head_dim)
        V = self.v_proj(x).view(N, self.num_heads, self.head_dim)

        pooled_list = []

        for b in torch.unique(batch_indices):
            mask = batch_indices == b
            Kb, Vb = K[mask], V[mask]  # [num_nodes_b, heads, head_dim]

            # Project seed vectors to Q
            Qb = self.q_proj(self.seed_vectors).view(
                self.num_seeds, self.num_heads, self.head_dim
            )
            # Scaled dot-product attention: [heads, num_seeds, num_nodes_b]
            attn_scores = torch.einsum("shd,nhd->hsn", Qb, Kb) / (self.head_dim**0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)

            # Weighted sum over nodes: [num_seeds, heads, head_dim]
            pooled_b = torch.einsum("hsn,nhd->shd", attn_probs, Vb)
            pooled_b = pooled_b.reshape(self.num_seeds, self.embed_dim)  # merge heads
            pooled_list.append(pooled_b)

        # Concatenate pooled embeddings from all groups
        pooled = torch.cat(pooled_list, dim=0)  # [num_groups*num_seeds, embed_dim]
        pooled = self.out_proj(pooled)
        return pooled

class TransformerBlock(nn.Module):
    """
    Standard Transformer block with multi-head self-attention and feedforward network.
    """
    def __init__(self, embed_dim, num_heads, ff_hidden_dim=None, dropout=0.0):
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = embed_dim * 2
        self.self_attn = SelfAttentionBlock(embed_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, batch_indices):
        # Multi-head self-attention
        attn_out = self.self_attn(x, batch_indices)
        x = self.norm1(x + attn_out)  # Residual connection and layer norm

        # Feedforward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual connection and layer norm
        return x


class PoolerTransformerBlock(nn.Module):
    """
    Transformer block with attention pooling and feedforward network.
    """

    def __init__(
        self, key_dim, num_heads, num_seeds=1, ff_hidden_dim=None, dropout=0.0
    ):
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = key_dim * 2
        self.attention_pool = AttentionPoolingBlock(
            key_dim, num_heads, num_seeds=num_seeds, dropout=dropout
        )
        self.norm1 = nn.LayerNorm(key_dim)
        self.ffn = nn.Sequential(
            nn.Linear(key_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, key_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(key_dim)

    def forward(self, x, batch_indices):
        # Attention pooling
        attn_out = self.attention_pool(x, batch_indices)
        x = self.norm1(attn_out)  # Layer norm

        # Feedforward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual connection and layer norm
        return x
