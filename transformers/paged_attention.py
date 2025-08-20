"""
When serving LLMs, the KV cache grows with context length (sequence length * hidden size * num layers). Naively, each request allocates a big contiguous chunk of GPU memory. With multiple requests of varying lengths, this leads to fragmentation and wasted memory.

Paged Attention solves this by:
- Breaking the KV cache into fixed-size pages (like virtual memory).
- Each request's cache is a list of page pointers, not one large block.
- During attention, gather KV from the right pages.

This enables efficient sharing, reuse, and avoids fragmentation → higher throughput, longer contexts.
"""
import torch

# toy settings
B, H, L, D = 2, 4, 16, 8  # batch, heads, sequence_len, dim
page_size = 4              # tokens per page

# KV cache broken into "pages"
num_pages = (L + page_size - 1) // page_size
K_pages = [torch.randn(H, page_size, D) for _ in range(num_pages)]
V_pages = [torch.randn(H, page_size, D) for _ in range(num_pages)]

# request mapping: token i -> (page_id, offset_in_page)
page_ids = torch.arange(L) // page_size
offsets  = torch.arange(L) % page_size

# gather full K,V for this sequence using page table
K_full = torch.stack([K_pages[p][..., o, :] for p, o in zip(page_ids, offsets)], dim=1)
V_full = torch.stack([V_pages[p][..., o, :] for p, o in zip(page_ids, offsets)], dim=1)

# attention
Q = torch.randn(B, H, 1, D)  # query for the next token
scores = torch.matmul(Q, K_full.transpose(-1, -2)) / (D ** 0.5)
attn = torch.softmax(scores, dim=-1)
out = torch.matmul(attn, V_full)

print(out.shape)  # (B, H, 1, D)
