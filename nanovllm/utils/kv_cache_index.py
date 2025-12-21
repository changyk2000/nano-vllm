import os
from threading import Event
from time import time
from typing import Optional

import numpy as np
import torch

from nanovllm.engine.sequence import Sequence

# Try to import kvikio for GPUDirect Storage support
try:
    import kvikio
    KVIKIO_AVAILABLE = True
except ImportError:
    KVIKIO_AVAILABLE = False


class KVCacheIndex:
    # TODO: cpu cache overflow ssd
    def __init__(self, gpu_kv_cache, index_name="imdb_kvcache.pt", use_gpudirect: bool = False) -> None:
        # Tensor[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.gpu_kv_cache = gpu_kv_cache

        self.save_dir = "/data/zhangyuyun/"
        self.index_name = index_name
        self.use_gpudirect = use_gpudirect and KVIKIO_AVAILABLE

        path = f"{self.save_dir}{self.index_name}"

        self.dirty = False
        self.indexed = False
        # text_id -> {
        #   "kv": Tensor[2, num_layers, seq_len_post_prune, num_kv_heads, head_dim],
        #   "pruning_len": int,                       # number of pruned text tokens
        #   "text_tokens_pruned": list[int] | None    # text tokens after pruning
        # }
        self.kv_cache_index: dict = {}
        
        # For GPUDirect: store file paths for each text_id's KV cache
        self.kv_cache_files: dict = {}
        
        if os.path.isfile(path):
            self.kv_cache_index = torch.load(path)
            self.indexed = True

            # pin memory when init
            for key, item in list(self.kv_cache_index.items()):
                # Support legacy tensor or new dict format
                if isinstance(item, torch.Tensor):
                    kv = item
                else:
                    kv = item.get("kv")
                if isinstance(kv, torch.Tensor) and not kv.is_pinned():
                    pinned_kv = kv.pin_memory()
                    if isinstance(item, torch.Tensor):
                        # Convert legacy tensor format to new dict format
                        self.kv_cache_index[key] = {"kv": pinned_kv}
                    else:
                        item["kv"] = pinned_kv
            
            # Load GPUDirect file mappings if available
            gds_files_path = f"{self.save_dir}{self.index_name}_gds_files.pt"
            if os.path.isfile(gds_files_path):
                self.kv_cache_files = torch.load(gds_files_path)
        
        if self.use_gpudirect:
            print("[KVCacheIndex] GPUDirect Storage enabled")
        elif use_gpudirect and not KVIKIO_AVAILABLE:
            print("[KVCacheIndex] GPUDirect requested but kvikio not available, falling back to standard transfer")

    def store_kv_cache(
        self,
        seqs: list[Sequence],
        stream: torch.cuda.Stream,
        return_timing: bool = False,
    ):
        _, num_layers, _, block_size, num_kv_heads, head_dim = self.gpu_kv_cache.shape
        with torch.cuda.stream(stream):
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)
            for seq in seqs:
                if seq.text_id is None or seq.text_id in self.kv_cache_index:
                    continue

                # Determine pruned indices (local prompt positions relative to text_token_len)
                pruned = sorted(seq.pruning_indices)
                # Allocate CPU cache for UNPRUNED tokens only
                pruned = [i for i in pruned if i < seq.text_token_len]
                post_prune_len = seq.text_token_len - len(pruned)
                cpu_kv_cache = torch.empty(
                    2,
                    num_layers,
                    post_prune_len,
                    num_kv_heads,
                    head_dim,
                    device="cpu",
                    dtype=self.gpu_kv_cache.dtype,
                    pin_memory=True,
                )

                # Build flattened slot list for the first text_token_len tokens
                token_slots: list[int] = []
                taken = 0
                for block_id in seq.block_table:
                    if taken >= seq.text_token_len:
                        break
                    # tokens available to take from this block
                    cnt = min(block_size, seq.text_token_len - taken)
                    base = block_id * block_size
                    token_slots.extend(range(base, base + cnt))
                    taken += cnt

                if len(token_slots) != seq.text_token_len:
                    # Fallback guard: lengths must match; otherwise skip indexing
                    token_slots = token_slots[: seq.text_token_len]

                # Filter out pruned positions to get kept slot ids
                if pruned:
                    kept_slots = [
                        slot for i, slot in enumerate(token_slots) if i not in pruned
                    ]
                else:
                    kept_slots = token_slots

                keep_len = len(kept_slots)
                # Compute pruned text token ids (final pruned state)
                kept_local_indices = [
                    i for i in range(seq.text_token_len) if i not in pruned
                ]
                text_tokens_pruned = [seq.token_ids[i] for i in kept_local_indices]

                assert keep_len != 0

                kept_slots_tensor = torch.tensor(
                    kept_slots, dtype=torch.int64, device=self.gpu_kv_cache.device
                )
                # Vectorized gather per (kv, layer), then one D2H copy per pair
                flat_blocks = self.gpu_kv_cache.shape[2] * block_size
                for kv_idx in range(2):
                    for layer_idx in range(num_layers):
                        src_flat = self.gpu_kv_cache[kv_idx, layer_idx].reshape(
                            flat_blocks, num_kv_heads, head_dim
                        )
                        selected = src_flat.index_select(0, kept_slots_tensor)
                        dst = cpu_kv_cache[kv_idx, layer_idx, :keep_len]
                        dst.copy_(selected, non_blocking=True)

                self.dirty = True
                self.kv_cache_index[seq.text_id] = {
                    "kv": cpu_kv_cache,
                    "pruning_len": len(pruned),
                    "text_tokens_pruned": text_tokens_pruned,
                }
                
                # Save GPUDirect file if enabled
                if self.use_gpudirect:
                    self._save_kv_cache_for_gpudirect(seq.text_id, cpu_kv_cache)
                    
        # No global synchronize here; let transfers overlap with subsequent work
        if self.dirty:
            event = torch.cuda.Event(blocking=False, enable_timing=return_timing)
            event.record(stream)
            if return_timing and start_event is not None:
                return event, start_event
            else:
                return event
        else:
            return None

    def get_kv_cache(
        self,
        seqs: list[Sequence],
        cancel_event: Event,
        stream: torch.cuda.Stream,
        return_timing: bool = False,
    ):
        """
        Schedule CPU->GPU H2D copies for KV cache on the provided CUDA stream.
        Returns a CUDA event recorded on that stream to signal completion, or None if no copies were enqueued.
        Does not call global synchronize.
        """
        _, num_layers, _, block_size, _, _ = self.gpu_kv_cache.shape
        any_copied = False
        # Use provided stream (preferred) or current stream
        with torch.cuda.stream(stream):
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)
            for seq in seqs:
                if seq.text_id is None or seq.num_cached_tokens >= seq.text_token_len:
                    continue
                item = self.kv_cache_index.get(seq.text_id)
                if item is None:
                    continue
                # Support legacy tensor or new dict format
                if isinstance(item, torch.Tensor):
                    cpu_kv_cache = item
                else:
                    cpu_kv_cache = item.get("kv")

                # Only copy tokens that aren't already cached (full blocks only)
                start_token = seq.num_cached_tokens
                start_block_idx = start_token // block_size
                token_offset = start_token
                # gpu_kv_cache[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                # cpu_kv_cache[2, num_layers, seq_len, num_kv_heads, head_dim]
                # Skip fully cached leading blocks
                for block_id in seq.block_table[start_block_idx:]:
                    if cancel_event.is_set():
                        break

                    remaining = seq.text_token_len - token_offset
                    if remaining <= 0:
                        break

                    block_tokens = remaining if remaining < block_size else block_size

                    for kv_idx in range(2):
                        for layer_idx in range(num_layers):
                            dst = self.gpu_kv_cache[
                                kv_idx, layer_idx, block_id, :block_tokens
                            ]
                            src = cpu_kv_cache[
                                kv_idx,
                                layer_idx,
                                token_offset : token_offset + block_tokens,
                            ]
                            # One copy: CPU -> GPU (non_blocking if src pinned)
                            dst.copy_(src, non_blocking=True)
                            any_copied = True

                    token_offset += block_tokens
                    seq.num_cached_tokens = token_offset

        if any_copied:
            event = torch.cuda.Event(blocking=False, enable_timing=return_timing)
            event.record(stream)
            if return_timing and start_event is not None:
                return event, start_event
            else:
                return event
        else:
            return None

    def is_indexed(self, seq: Sequence):
        return seq.text_id and seq.text_id in self.kv_cache_index

    def get_kv_cache_gpudirect(
        self,
        seqs: list[Sequence],
        cancel_event: Event,
        stream: torch.cuda.Stream,
        return_timing: bool = False,
    ):
        """
        Load KV cache from SSD directly to GPU using GPUDirect Storage (kvikio).
        Falls back to standard transfer if GPUDirect is not available or fails.
        Returns a CUDA event recorded on that stream to signal completion, or None if no copies were enqueued.
        """
        if not self.use_gpudirect or not KVIKIO_AVAILABLE:
            return self.get_kv_cache(seqs, cancel_event, stream, return_timing)

        _, num_layers, _, block_size, _, _ = self.gpu_kv_cache.shape
        any_copied = False

        with torch.cuda.stream(stream):
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)

            for seq in seqs:
                if seq.text_id is None or seq.num_cached_tokens >= seq.text_token_len:
                    continue
                item = self.kv_cache_index.get(seq.text_id)
                if item is None:
                    continue
                
                # Get the KV cache file path for GPUDirect loading
                kv_file_path = self.kv_cache_files.get(seq.text_id)
                
                if kv_file_path and os.path.isfile(kv_file_path):
                    # Use GPUDirect to load directly from SSD to GPU
                    try:
                        # Get shape info from the index item
                        # Support legacy tensor or new dict format
                        if isinstance(item, torch.Tensor):
                            cpu_kv_cache = item
                        else:
                            cpu_kv_cache = item.get("kv")
                        if cpu_kv_cache is None:
                            continue
                        
                        # Allocate GPU tensor for direct loading with same shape
                        gpu_tensor = torch.empty(
                            cpu_kv_cache.shape,
                            dtype=cpu_kv_cache.dtype,
                            device=self.gpu_kv_cache.device
                        )
                        
                        # Use GPUDirect to read directly from SSD to GPU memory
                        with kvikio.CuFile(kv_file_path, "r") as f:
                            f.read(gpu_tensor)
                        
                        # Copy to the actual KV cache positions
                        start_token = seq.num_cached_tokens
                        start_block_idx = start_token // block_size
                        token_offset = start_token
                        
                        for block_id in seq.block_table[start_block_idx:]:
                            if cancel_event.is_set():
                                break

                            remaining = seq.text_token_len - token_offset
                            if remaining <= 0:
                                break

                            block_tokens = remaining if remaining < block_size else block_size

                            for kv_idx in range(2):
                                for layer_idx in range(num_layers):
                                    dst = self.gpu_kv_cache[
                                        kv_idx, layer_idx, block_id, :block_tokens
                                    ]
                                    src = gpu_tensor[
                                        kv_idx,
                                        layer_idx,
                                        token_offset : token_offset + block_tokens,
                                    ]
                                    dst.copy_(src, non_blocking=True)
                                    any_copied = True

                            token_offset += block_tokens
                            seq.num_cached_tokens = token_offset
                    except Exception as e:
                        print(f"[GPUDirect] Failed to load KV cache for {seq.text_id}: {e}, falling back to standard transfer")
                        # Fallback to standard CPU->GPU transfer
                        # Support legacy tensor or new dict format
                        if isinstance(item, torch.Tensor):
                            cpu_kv_cache = item
                        else:
                            cpu_kv_cache = item.get("kv")
                        if cpu_kv_cache is not None:
                            self._copy_kv_from_cpu(seq, cpu_kv_cache, cancel_event, block_size, num_layers)
                            any_copied = True
                else:
                    # No GPUDirect file, use standard CPU->GPU transfer
                    # Support legacy tensor or new dict format
                    if isinstance(item, torch.Tensor):
                        cpu_kv_cache = item
                    else:
                        cpu_kv_cache = item.get("kv")
                    if cpu_kv_cache is not None:
                        self._copy_kv_from_cpu(seq, cpu_kv_cache, cancel_event, block_size, num_layers)
                        any_copied = True

        if any_copied:
            event = torch.cuda.Event(blocking=False, enable_timing=return_timing)
            event.record(stream)
            if return_timing and start_event is not None:
                return event, start_event
            else:
                return event
        else:
            return None

    def _copy_kv_from_cpu(
        self,
        seq: Sequence,
        cpu_kv_cache: torch.Tensor,
        cancel_event: Event,
        block_size: int,
        num_layers: int,
    ):
        """Helper method to copy KV cache from CPU to GPU."""
        start_token = seq.num_cached_tokens
        start_block_idx = start_token // block_size
        token_offset = start_token

        for block_id in seq.block_table[start_block_idx:]:
            if cancel_event.is_set():
                break

            remaining = seq.text_token_len - token_offset
            if remaining <= 0:
                break

            block_tokens = remaining if remaining < block_size else block_size

            for kv_idx in range(2):
                for layer_idx in range(num_layers):
                    dst = self.gpu_kv_cache[
                        kv_idx, layer_idx, block_id, :block_tokens
                    ]
                    src = cpu_kv_cache[
                        kv_idx,
                        layer_idx,
                        token_offset : token_offset + block_tokens,
                    ]
                    dst.copy_(src, non_blocking=True)

            token_offset += block_tokens
            seq.num_cached_tokens = token_offset

    def _save_kv_cache_for_gpudirect(self, text_id: int, kv_tensor: torch.Tensor):
        """
        Save KV cache tensor to a separate binary file for GPUDirect loading.
        """
        if not KVIKIO_AVAILABLE:
            return
        
        kv_dir = os.path.join(self.save_dir, "kv_cache_gds")
        os.makedirs(kv_dir, exist_ok=True)
        
        file_path = os.path.join(kv_dir, f"kv_{text_id}.bin")
        
        try:
            # Ensure tensor is contiguous for direct IO
            if not kv_tensor.is_contiguous():
                kv_tensor = kv_tensor.contiguous()
            
            # Write tensor data directly
            with kvikio.CuFile(file_path, "w") as f:
                # If tensor is on CPU, move to GPU for kvikio write
                if kv_tensor.device.type == "cpu":
                    gpu_tensor = kv_tensor.to(self.gpu_kv_cache.device)
                    f.write(gpu_tensor)
                else:
                    f.write(kv_tensor)
            
            self.kv_cache_files[text_id] = file_path
        except Exception as e:
            print(f"[GPUDirect] Failed to save KV cache for {text_id}: {e}")

    def persistence(self):
        print("[persistence]")
        if self.dirty:
            path = f"{self.save_dir}{self.index_name}"
            torch.save(self.kv_cache_index, path)
            
            # Also save GPUDirect file paths
            if self.use_gpudirect and self.kv_cache_files:
                files_path = f"{self.save_dir}{self.index_name}_gds_files.pt"
                torch.save(self.kv_cache_files, files_path)
        self.dirty = False


# Triton can only load GPU memory, so useless for now

# @triton.jit
# def get_kv_cache_kernel(
#     cpu_key_ptr,
#     cpu_value_ptr,
#     gpu_key_ptr,
#     gpu_value_ptr,
#     slot_mapping_ptr,
#     cpu_layer_stride,
#     gpu_layer_stride,
#     num_layers: tl.constexpr,
#     D: tl.constexpr,
# ):
#     idx = tl.program_id(0)
#     slot = tl.load(slot_mapping_ptr + idx)
#     if slot == -1:
#         return
#     for i in range(num_layers):
#         # offsets for source (key/value) rows
#         cpu_offsets = i * cpu_layer_stride + idx * D + tl.arange(0, D)
#         key = tl.load(cpu_key_ptr + cpu_offsets)
#         value = tl.load(cpu_value_ptr + cpu_offsets)
#
#         # offsets for destination cache (flattened)
#         gpu_offsets = i * gpu_layer_stride + slot * D + tl.arange(0, D)
#         tl.store(gpu_key_ptr + gpu_offsets, key)
#         tl.store(gpu_value_ptr + gpu_offsets, value)
#
#
# def get_kv_cache(
#     gpu_key_cache: torch.Tensor,
#     gpu_value_cache: torch.Tensor,
#     cpu_key_cache: torch.Tensor,
#     cpu_value_cache: torch.Tensor,
#     slot_mapping: torch.Tensor,
# ):
#     # get cpu to gpu
#     # gpu_key, gpu_value: [num_layers, num_blocks * block_size, num_heads, head_dim]
#     # cpu_key, cpu_value: [num_layers, seq_len, num_heads, head_dim]
#
#     num_layers, _, num_heads, head_dim = gpu_key_cache.shape
#     _, seq_len, num_heads, head_dim = cpu_key_cache.shape
#
#     D = num_heads * head_dim
#     assert slot_mapping.numel() == seq_len
#     # Launch one program per token for every layers to scatter into cache
#     get_kv_cache_kernel[(seq_len,)](
#         cpu_key_cache,
#         cpu_value_cache,
#         gpu_key_cache,
#         gpu_value_cache,
#         slot_mapping,
#         cpu_key_cache.stride(0),
#         gpu_key_cache.stride(0),
#         num_layers,  # type: ignore
#         D,  # type: ignore
#     )
