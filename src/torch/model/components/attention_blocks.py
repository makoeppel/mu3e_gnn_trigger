import torch
import torch.nn as nn
import torch.nn.functional as F


# Alternative implementation using torch.nn.functional.scaled_dot_product_attention
# (Available in PyTorch 2.0+, provides additional optimizations)
class SelfAttentionBlock(nn.Module):
    """
    Optimized version using PyTorch's built-in scaled_dot_product_attention.
    Requires PyTorch 2.0+ but provides maximum efficiency.
    """

    def __init__(self, embed_dim, num_heads, dropout=0.0):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout_p = dropout

        # Cache for attention mask
        self._cached_mask = None
        self._cached_batch_indices = None

    def _create_attention_mask(self, batch_indices):
        """Create attention mask for SDPA (additive format)."""
        if self._cached_batch_indices is not None and torch.equal(
            batch_indices, self._cached_batch_indices
        ):
            return self._cached_mask

        N = batch_indices.size(0)
        # Create boolean mask: True where attention is allowed (same batch)
        mask = batch_indices.unsqueeze(0) == batch_indices.unsqueeze(1)  # [N, N]

        # Convert to additive mask format for SDPA
        # Use 0.0 for allowed positions, -inf for masked positions
        attn_mask = torch.zeros_like(
            mask, dtype=torch.float, device=batch_indices.device
        )
        attn_mask.masked_fill_(~mask, float("-inf"))

        self._cached_mask = attn_mask
        self._cached_batch_indices = batch_indices.clone()

        return attn_mask

    def forward(self, x, batch_indices):
        """
        Args:
            x: [N, embed_dim] node embeddings
            batch_indices: [N] tensor with graph/set ids for each node
        Returns:
            out: [N, embed_dim] updated node embeddings
        """
        N, _ = x.size()

        # Project and reshape for multi-head attention
        qkv = self.qkv_proj(x).view(N, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(1)  # each [N, num_heads, head_dim]

        # Reshape to [batch_size=1, num_heads, seq_len, head_dim] for SDPA
        # SDPA expects batch dimension first
        q = q.unsqueeze(0).transpose(1, 2)  # [1, num_heads, N, head_dim]
        k = k.unsqueeze(0).transpose(1, 2)  # [1, num_heads, N, head_dim]
        v = v.unsqueeze(0).transpose(1, 2)  # [1, num_heads, N, head_dim]

        # Create attention mask
        attn_mask = self._create_attention_mask(batch_indices)  # [N, N]

        # Use built-in scaled dot product attention
        # Note: SDPA can be sensitive to mask format and tensor layouts
        out = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout_p if self.training else 0.0,
            is_causal=False,
        )
        # Reshape output: [1, num_heads, N, head_dim] -> [N, embed_dim]
        out = out.squeeze(0).transpose(1, 2).contiguous().view(N, self.embed_dim)
        out = self.out_proj(out)

        return out


class AttentionPoolingBlock(nn.Module):
    """
    Optimized version using PyTorch's scaled_dot_product_attention.
    Uses attention masks instead of loops for maximum efficiency.
    """

    def __init__(
        self,
        embed_dim,
        num_heads,
        num_seeds,
        dropout=0.0,
        seed_init="normal",
        seed_std=0.02,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_seeds = num_seeds
        self.head_dim = embed_dim // num_heads

        # Learnable seed vectors
        self.seed_vectors = nn.Parameter(torch.empty(num_seeds, embed_dim))

        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.kv_proj = nn.Linear(embed_dim, 2 * embed_dim, bias=False)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout

        # Initialize seeds
        self._init_seeds(seed_init, seed_std)

        # Cache for attention mask
        self._cached_mask = None
        self._cached_batch_indices = None
        self._cached_num_batches = None

    def _init_seeds(self, init_method, std):
        if init_method == "normal":
            nn.init.normal_(self.seed_vectors, mean=0.0, std=std)
        elif init_method == "xavier":
            nn.init.xavier_uniform_(self.seed_vectors)
        elif init_method == "zeros":
            nn.init.zeros_(self.seed_vectors)
        else:
            raise ValueError(f"Unknown initialization method: {init_method}")

    def _create_pooling_attention_mask(self, batch_indices):
        """Create boolean attention mask for SDPA."""
        if self._cached_batch_indices is not None and torch.equal(
            batch_indices, self._cached_batch_indices
        ):
            return self._cached_mask, self._cached_num_batches

        unique_batches = torch.unique(batch_indices, sorted=True)
        B = len(unique_batches)
        N = batch_indices.size(0)

        # Create mapping from batch_id to batch_index
        batch_id_to_idx = {
            batch_id.item(): idx for idx, batch_id in enumerate(unique_batches)
        }
        batch_idx_tensor = torch.tensor(
            [batch_id_to_idx[bid.item()] for bid in batch_indices],
            device=batch_indices.device,
        )

        # Create boolean mask for SDPA
        seed_batch_indices = torch.arange(
            B, device=batch_indices.device
        ).repeat_interleave(self.num_seeds)
        mask = seed_batch_indices.unsqueeze(1) == batch_idx_tensor.unsqueeze(
            0
        )  # [B * num_seeds, N]

        self._cached_mask = mask
        self._cached_batch_indices = batch_indices.clone()
        self._cached_num_batches = B

        return mask, B

    def forward(self, x, batch_indices):
        """
        Args:
            x: [N, embed_dim] input embeddings
            batch_indices: [N] batch indices

        Returns:
            pooled: [B, num_seeds, embed_dim] pooled representations
        """
        N, _ = x.size()

        # Create attention mask
        attn_mask, B = self._create_pooling_attention_mask(batch_indices)

        # Prepare queries from seed vectors
        queries = (
            self.seed_vectors.unsqueeze(0)
            .expand(B, -1, -1)
            .contiguous()
            .view(B * self.num_seeds, self.embed_dim)
        )
        q = self.q_proj(queries)

        # Project keys and values
        kv = self.kv_proj(x)
        k, v = kv.chunk(2, dim=-1)

        # Reshape for multi-head attention
        q = q.view(B * self.num_seeds, self.num_heads, self.head_dim).transpose(0, 1)
        k = k.view(N, self.num_heads, self.head_dim).transpose(0, 1)
        v = v.view(N, self.num_heads, self.head_dim).transpose(0, 1)

        # Use scaled dot product attention with mask
        pooled = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=False,
        )  # [num_heads, B * num_seeds, head_dim]

        # Reshape output
        pooled = (
            pooled.transpose(0, 1).contiguous().view(B * self.num_seeds, self.embed_dim)
        )
        pooled = pooled.view(B, self.num_seeds, self.embed_dim)

        # Output projection
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
