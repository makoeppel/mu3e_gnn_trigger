import torch
import torch.nn as nn
import torch.nn.functional as F


class SelfAttentionBlock(nn.Module):
    """
    Multi-head self-attention that computes attention per group defined by batch indices.
    Similar to torch.nn.MultiheadAttention, but supports variable-length groups using batch indices.
    """

    def __init__(self, key_dim, num_heads, dropout=0.0):
        super().__init__()
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads

        # Linear projections
        self.q_proj = nn.ModuleList(
            [nn.Linear(key_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(key_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(key_dim, self.head_dim) for _ in range(num_heads)]
        )

        # Final output projection
        self.out_proj = nn.Linear(key_dim, key_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, batch_indices):
        """
        Args:
            x: [N, embed_dim] node or graph embeddings
            batch_indices: [N] integer tensor indicating group membership
        Returns:
            out: [N, embed_dim] same shape as input
        """
        N, C = x.size()

        # Project inputs to Q, K, V for each head
        Q = torch.stack(
            [proj(x) for proj in self.q_proj], dim=1
        )  # [N, heads, head_dim]
        K = torch.stack(
            [proj(x) for proj in self.k_proj], dim=1
        )  # [N, heads, head_dim]
        V = torch.stack(
            [proj(x) for proj in self.v_proj], dim=1
        )  # [N, heads, head_dim]

        # Prepare output
        out = torch.zeros_like(Q)

        # Process each batch separately
        unique_batches = torch.unique(batch_indices)
        for b in unique_batches:
            mask = batch_indices == b
            Qb, Kb, Vb = Q[mask], K[mask], V[mask]  # [num_nodes_b, heads, head_dim]

            # Compute attention scores: [heads, num_nodes_b, num_nodes_b]
            attn_scores = torch.einsum("nhd,mhd->hnm", Qb, Kb) / (Kb.shape[0] ** 0.5)
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            # Weighted sum: [num_nodes_b, heads, head_dim]
            out_b = torch.einsum("hnm,mhd->nhd", attn_probs, Vb)
            out[mask] = out_b

        # Merge heads
        out = out.contiguous().view(N, self.key_dim)  # [N, key_dim]
        # Final linear projection
        out = self.out_proj(out)
        return out


class AttentionPoolingBlock(nn.Module):
    """
    Multi-head attention pooling per group using batch indices.
    Each head has its own projection (key_dim), like your SelfAttentionBlock.
    Outputs one pooled embedding per group.
    """

    def __init__(self, key_dim, num_heads, num_seeds=1, dropout=0.0):
        super().__init__()
        assert key_dim % num_heads == 0, "key_dim must be divisible by num_heads"
        self.key_dim = key_dim
        self.num_heads = num_heads
        self.head_dim = key_dim // num_heads

        # One linear per head
        self.q_proj = nn.ModuleList(
            [nn.Linear(key_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.k_proj = nn.ModuleList(
            [nn.Linear(key_dim, self.head_dim) for _ in range(num_heads)]
        )
        self.v_proj = nn.ModuleList(
            [nn.Linear(key_dim, self.head_dim) for _ in range(num_heads)]
        )

        self.out_proj = nn.Linear(self.key_dim, self.key_dim)
        self.dropout = nn.Dropout(dropout)

        # Optional: learnable seed per head (can pool via query)
        self.seed_vectors = nn.Parameter(torch.randn(num_seeds, self.key_dim))
        self.num_seeds = num_seeds

    def forward(self, x, batch_indices):
        """
        Args:
            x: [N, embed_dim] node embeddings
            batch_indices: [N] integer tensor indicating group membership
        Returns:
            pooled: [num_groups, embed_dim] pooled embedding per group
        """
        N, C = x.size()
        device = x.device

        # Project K and V per head
        K = torch.stack(
            [proj(x) for proj in self.k_proj], dim=1
        )  # [N, heads, head_dim]
        V = torch.stack(
            [proj(x) for proj in self.v_proj], dim=1
        )  # [N, heads, head_dim]

        pooled_list = []
        unique_batches = torch.unique(batch_indices)
        for b in unique_batches:
            mask = batch_indices == b
            Kb, Vb = K[mask], V[mask]  # [num_nodes_b, heads, head_dim]

            # Q = learnable seed vectors, split by head
            Qb = torch.stack(
                [proj(self.seed_vectors) for h, proj in enumerate(self.q_proj)], dim=1
            )  # [num_seeds, heads, head_dim]

            # Attention: [heads, num_seeds, num_nodes_b]
            attn_scores = torch.einsum("shd,nhd->hsn", Qb, Kb) / (
                self.num_seeds**0.5
            )  # Scaled dot-product
            attn_probs = F.softmax(attn_scores, dim=-1)  # Softmax over nodes
            attn_probs = self.dropout(attn_probs)  # [heads, num_seeds, num_nodes_b]

            # Weighted sum: [1, heads, head_dim]
            pooled_b = torch.einsum("hsn,nhd->shd", attn_probs, Vb)  # Pool over nodes
            pooled_b = pooled_b.contiguous().view(
                self.num_seeds, self.key_dim
            )  # [num_seeds, embed_dim]
            pooled_list.append(pooled_b)  # List of [num_seeds, embed_dim]

        # Concatenate pooled embeddings per group
        pooled = torch.cat(pooled_list, dim=0)  # [num_groups, embed_dim]
        pooled = self.out_proj(pooled)  # Final linear projection
        return pooled  # [num_groups, embed_dim]


class TransformerBlock(nn.Module):
    """
    Transformer block with multi-head self-attention and feedforward network.
    """

    def __init__(self, key_dim, num_heads, ff_hidden_dim = None, dropout=0.0):
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = key_dim * 2
        self.attention = SelfAttentionBlock(key_dim, num_heads, dropout)
        self.norm1 = nn.LayerNorm(key_dim)
        self.ffn = nn.Sequential(
            nn.Linear(key_dim, ff_hidden_dim),
            nn.ReLU(),
            nn.Linear(ff_hidden_dim, key_dim),
            nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(key_dim)

    def forward(self, x, batch_indices):
        # Multi-head self-attention
        attn_out = self.attention(x, batch_indices)
        x = self.norm1(x + attn_out)  # Residual connection and layer norm

        # Feedforward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)  # Residual connection and layer norm
        return x


class PoolerTransformerBlock(nn.Module):
    """
    Transformer block with attention pooling and feedforward network.
    """

    def __init__(self, key_dim, num_heads, num_seeds = 1, ff_hidden_dim=None, dropout=0.0):
        super().__init__()
        if ff_hidden_dim is None:
            ff_hidden_dim = key_dim * 2
        self.attention_pool = AttentionPoolingBlock(key_dim, num_heads, num_seeds=num_seeds, dropout = dropout)
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
