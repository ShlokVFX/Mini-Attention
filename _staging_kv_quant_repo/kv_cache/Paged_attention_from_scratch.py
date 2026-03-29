"Run the file with : uv run python kernels/Kv_Cache/Paged_attention_from_scratch.py"

import torch
import math


class Page:
    def __init__(self, page_size, num_heads, head_dim, dtype=torch.float32, device="cpu"):
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.keys = torch.zeros(page_size, num_heads, head_dim, dtype=dtype, device=device)
        self.values = torch.zeros(page_size, num_heads, head_dim, dtype=dtype, device=device)
        self.ref_count = 0

    def update_access(self):
        self.ref_count += 1


class PageTable:
    def __init__(self):
        self.logical_to_physical = {}

    def map_page(self, logical_page_id, physical_page_id):
        self.logical_to_physical[logical_page_id] = physical_page_id

    def get_physical_page(self, logical_page_id):
        return self.logical_to_physical.get(logical_page_id, -1)


class BlockManager:
    def __init__(self, num_pages, page_size, num_heads, head_dim, dtype=torch.float32, device="cpu"):
        self.num_pages = num_pages
        self.page_size = page_size
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.pages = [
            Page(page_size, num_heads, head_dim, dtype=dtype, device=device)
            for _ in range(num_pages)
        ]
        self.free_pages = list(range(num_pages))
        self.allocated_pages = set()

    def allocate_page(self):
        if not self.free_pages:
            raise RuntimeError("No free pages available")
        page_id = self.free_pages.pop()
        self.allocated_pages.add(page_id)
        self.pages[page_id].ref_count += 1
        return page_id

    def free_page(self, page_id):
        if page_id in self.allocated_pages:
            self.pages[page_id].ref_count -= 1
            if self.pages[page_id].ref_count <= 0:
                self.pages[page_id].ref_count = 0
                self.allocated_pages.remove(page_id)
                self.free_pages.append(page_id)


class SequenceManager:
    def __init__(self, block_manager):
        self.block_manager = block_manager
        self.sequences = {}

    def create_sequence(self, seq_id):
        page_table = PageTable()
        self.sequences[seq_id] = page_table
        return page_table

    def append_token(self, seq_id, token_pos, key, value):
        if seq_id not in self.sequences:
            raise KeyError(f"Sequence {seq_id} does not exist")

        page_table = self.sequences[seq_id]

        logical_page_id = token_pos // self.block_manager.page_size
        page_offset = token_pos % self.block_manager.page_size

        physical_page_id = page_table.get_physical_page(logical_page_id)
        if physical_page_id == -1:
            physical_page_id = self.block_manager.allocate_page()
            page_table.map_page(logical_page_id, physical_page_id)

        page = self.block_manager.pages[physical_page_id]

        if key.shape != (self.block_manager.num_heads, self.block_manager.head_dim):
            raise ValueError("Key shape mismatch")
        if value.shape != (self.block_manager.num_heads, self.block_manager.head_dim):
            raise ValueError("Value shape mismatch")

        page.keys[page_offset].copy_(key)
        page.values[page_offset].copy_(value)
        page.update_access()


class PagedAttention:
    def __init__(self, block_manager):
        self.block_manager = block_manager

    def compute_attention(self, query, page_table, seq_len):
        if seq_len == 0:
            return torch.zeros_like(query)

        batch_size, num_heads, head_dim = query.shape
        scale = 1.0 / math.sqrt(head_dim)

        all_keys = []
        all_values = []

        num_pages = (seq_len + self.block_manager.page_size - 1) // self.block_manager.page_size

        for logical_page_id in range(num_pages):
            physical_page_id = page_table.get_physical_page(logical_page_id)
            if physical_page_id == -1:
                continue

            page = self.block_manager.pages[physical_page_id]
            start_idx = logical_page_id * self.block_manager.page_size
            valid_tokens = min(self.block_manager.page_size, seq_len - start_idx)

            if valid_tokens > 0:
                all_keys.append(page.keys[:valid_tokens])
                all_values.append(page.values[:valid_tokens])

        if len(all_keys) == 0:
            return torch.zeros_like(query)

        keys = torch.cat(all_keys, dim=0)      # [S, H, D]
        values = torch.cat(all_values, dim=0)  # [S, H, D]

        S = keys.shape[0]

        keys = keys.permute(1, 0, 2)           # [H, S, D]
        values = values.permute(1, 0, 2)       # [H, S, D]

        query = query                          # [B, H, D]

        scores = torch.einsum("bhd,hsd->bhs", query, keys) * scale  # [B, H, S]
        attn = torch.softmax(scores, dim=-1)                        # [B, H, S]
        output = torch.einsum("bhs,hsd->bhd", attn, values)          # [B, H, D]

        return output


if __name__ == "__main__":
    torch.manual_seed(0)

    num_pages = 100
    page_size = 8
    num_heads = 4
    head_dim = 16
    batch_size = 1

    block_manager = BlockManager(num_pages, page_size, num_heads, head_dim)
    seq_manager = SequenceManager(block_manager)
    paged_attn = PagedAttention(block_manager)

    seq_id = 0
    page_table = seq_manager.create_sequence(seq_id)
    seq_len = 0

    print("Starting simulation of generating 20 tokens...")

    for token_pos in range(20):
        key = torch.randn(num_heads, head_dim)
        value = torch.randn(num_heads, head_dim)

        seq_manager.append_token(seq_id, token_pos, key, value)
        seq_len += 1

        if (token_pos + 1) % 5 == 0:
            query = torch.randn(batch_size, num_heads, head_dim)
            output = paged_attn.compute_attention(query, page_table, seq_len)

            used_pages = len(page_table.logical_to_physical)

            print(f"Processed Tokens: {token_pos + 1}")
            print(f"Attention output shape: {output.shape}")
            print(f"Currently used physical pages: {used_pages}")
            print("-" * 50)

