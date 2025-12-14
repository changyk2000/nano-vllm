import os
from threading import Event
from typing import Optional

import numpy as np
import torch

from nanovllm.engine.sequence import Sequence


class KVCacheIndex:
    """
    KV缓存索引类：用于管理和持久化键值（Key-Value）缓存
    
    该类实现了一个KV缓存索引系统，用于：
    1. 将GPU上的KV缓存转移到CPU内存中进行持久化存储
    2. 在需要时将CPU内存中的KV缓存恢复到GPU
    3. 支持剪枝（pruning）后的KV缓存存储
    
    主要应用场景：
    - 对于长文本，可以预先计算并缓存其KV值
    - 在多次推理中重用这些缓存，避免重复计算
    - 支持对文本进行剪枝后，只缓存保留的token对应的KV值
    """
    # TODO: cpu cache overflow ssd
    # TODO: CPU缓存溢出时转存到SSD
    
    def __init__(self, gpu_kv_cache, index_name="imdb_kvcache.pt") -> None:
        """
        初始化KV缓存索引
        
        参数:
            gpu_kv_cache: GPU上的KV缓存张量
                形状: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                2代表Key和Value两个缓存
            index_name: 索引文件名，用于持久化存储
        """
        # GPU上的KV缓存，形状: Tensor[2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
        self.gpu_kv_cache = gpu_kv_cache

        # 保存目录和索引文件名
        self.save_dir = "/data/zwt/"
        self.index_name = index_name

        path = f"{self.save_dir}{self.index_name}"

        # dirty标志：标记索引是否有未保存的修改
        self.dirty = False
        # indexed标志：标记是否已从磁盘加载了索引
        self.indexed = False
        
        # KV缓存索引字典：text_id -> 缓存数据
        # 数据结构:
        # text_id -> {
        #   "kv": Tensor[2, num_layers, seq_len_post_prune, num_kv_heads, head_dim],
        #         # 剪枝后的KV缓存，存储在CPU内存（pinned memory）中
        #   "pruning_len": int,                       # 被剪枝的token数量
        #   "text_tokens_pruned": list[int] | None    # 剪枝后保留的token ID列表
        # }
        self.kv_cache_index: dict = {}
        
        # 如果索引文件存在，则从磁盘加载
        if os.path.isfile(path):
            self.kv_cache_index = torch.load(path)
            self.indexed = True

            # 初始化时将所有KV缓存固定在内存中（pin memory）
            # 固定内存可以加速CPU到GPU的数据传输
            for _, item in list(self.kv_cache_index.items()):
                kv = item.get("kv")
                if isinstance(kv, torch.Tensor) and not kv.is_pinned():
                    item["kv"] = kv.pin_memory()

    def store_kv_cache(
        self,
        seqs: list[Sequence],
        stream: torch.cuda.Stream,
        return_timing: bool = False,
    ):
        """
        将序列的KV缓存从GPU转移到CPU内存中存储
        
        该方法的核心功能：
        1. 从GPU的KV缓存中提取指定序列的KV值
        2. 处理剪枝（pruning）：只保存未被剪枝的token对应的KV值
        3. 将KV值异步传输到CPU的pinned memory中
        4. 更新索引字典
        
        参数:
            seqs: 需要存储KV缓存的序列列表
            stream: CUDA流，用于异步传输
            return_timing: 是否返回计时信息
            
        返回:
            如果有数据传输，返回CUDA事件用于同步；否则返回None
        """
        _, num_layers, _, block_size, num_kv_heads, head_dim = self.gpu_kv_cache.shape
        with torch.cuda.stream(stream):
            # 如果需要计时，创建开始事件
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)
            
            # 遍历每个序列
            for seq in seqs:
                # 跳过以下情况：
                # 1. 没有text_id的序列（无法索引）
                # 2. 已经在索引中的序列（避免重复存储）
                if seq.text_id is None or seq.text_id in self.kv_cache_index:
                    continue

                # 确定被剪枝的位置索引（相对于文本token的本地位置）
                pruned = sorted(seq.pruning_indices)
                # 只保留文本部分的剪枝索引（不包括任务提示部分）
                pruned = [i for i in pruned if i < seq.text_token_len]
                # 计算剪枝后剩余的token数量
                post_prune_len = seq.text_token_len - len(pruned)
                
                # 在CPU上分配pinned memory用于存储未被剪枝的token的KV缓存
                # 使用pinned memory可以加速后续的H2D（CPU到GPU）传输
                cpu_kv_cache = torch.empty(
                    2,  # Key和Value
                    num_layers,
                    post_prune_len,  # 剪枝后的长度
                    num_kv_heads,
                    head_dim,
                    device="cpu",
                    dtype=self.gpu_kv_cache.dtype,
                    pin_memory=True,  # 固定内存
                )

                # 构建前text_token_len个token的slot列表
                # slot是KV缓存中的物理位置索引
                token_slots: list[int] = []
                taken = 0
                for block_id in seq.block_table:
                    if taken >= seq.text_token_len:
                        break
                    # 计算从当前block中可以取多少个token
                    cnt = min(block_size, seq.text_token_len - taken)
                    base = block_id * block_size
                    # 将这些slot添加到列表中
                    token_slots.extend(range(base, base + cnt))
                    taken += cnt

                # 保护措施：如果长度不匹配，截断到正确长度
                if len(token_slots) != seq.text_token_len:
                    token_slots = token_slots[: seq.text_token_len]

                # 过滤掉被剪枝位置的slot，得到保留的slot列表
                if pruned:
                    kept_slots = [
                        slot for i, slot in enumerate(token_slots) if i not in pruned
                    ]
                else:
                    kept_slots = token_slots

                keep_len = len(kept_slots)
                
                # 计算剪枝后的文本token ID列表
                kept_local_indices = [
                    i for i in range(seq.text_token_len) if i not in pruned
                ]
                text_tokens_pruned = [seq.token_ids[i] for i in kept_local_indices]

                assert keep_len != 0

                # 将kept_slots转换为GPU上的张量，用于后续的索引操作
                kept_slots_tensor = torch.tensor(
                    kept_slots, dtype=torch.int64, device=self.gpu_kv_cache.device
                )
                
                # 对每一对(kv, layer)进行向量化gather操作，然后执行一次D2H（GPU到CPU）拷贝
                flat_blocks = self.gpu_kv_cache.shape[2] * block_size
                for kv_idx in range(2):  # Key和Value
                    for layer_idx in range(num_layers):
                        # 将GPU缓存reshape为扁平化格式 [flat_blocks, num_kv_heads, head_dim]
                        src_flat = self.gpu_kv_cache[kv_idx, layer_idx].reshape(
                            flat_blocks, num_kv_heads, head_dim
                        )
                        # 使用kept_slots_tensor从GPU缓存中选择需要的KV值
                        selected = src_flat.index_select(0, kept_slots_tensor)
                        # 获取CPU缓存中对应的目标位置
                        dst = cpu_kv_cache[kv_idx, layer_idx, :keep_len]
                        # 异步拷贝：GPU -> CPU（non_blocking=True）
                        dst.copy_(selected, non_blocking=True)

                # 标记索引已被修改，需要持久化
                self.dirty = True
                # 将KV缓存和元数据存入索引字典
                self.kv_cache_index[seq.text_id] = {
                    "kv": cpu_kv_cache,  # CPU上的KV缓存
                    "pruning_len": len(pruned),  # 剪枝的token数量
                    "text_tokens_pruned": text_tokens_pruned,  # 剪枝后的token列表
                }
        # 不在这里进行全局同步；让传输与后续工作并行执行
        if self.dirty:
            # 创建事件记录传输完成时间点
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
        从CPU内存异步加载KV缓存到GPU
        
        该方法的核心功能：
        1. 在指定的CUDA流上调度CPU到GPU的H2D（Host to Device）拷贝
        2. 只拷贝尚未缓存的部分（基于num_cached_tokens）
        3. 支持取消操作（通过cancel_event）
        4. 不进行全局同步，允许与其他操作并行
        
        参数:
            seqs: 需要加载KV缓存的序列列表
            cancel_event: 取消事件，用于中断加载操作
            stream: CUDA流，用于异步传输
            return_timing: 是否返回计时信息
            
        返回:
            如果有拷贝操作，返回记录在该流上的CUDA事件；否则返回None
        """
        _, num_layers, _, block_size, _, _ = self.gpu_kv_cache.shape
        any_copied = False
        # 使用提供的stream（首选）或当前stream
        with torch.cuda.stream(stream):
            start_event = (
                torch.cuda.Event(enable_timing=True) if return_timing else None
            )
            if start_event is not None:
                start_event.record(stream)
            
            for seq in seqs:
                # 跳过以下情况：
                # 1. 没有text_id的序列（没有可加载的缓存）
                # 2. 已经完全缓存的序列（num_cached_tokens >= text_token_len）
                if seq.text_id is None or seq.num_cached_tokens >= seq.text_token_len:
                    continue
                
                # 从索引中获取该序列的KV缓存
                item = self.kv_cache_index.get(seq.text_id)
                if item is None:
                    continue
                
                # 支持旧版本的张量格式或新的字典格式
                cpu_kv_cache = item.get("kv")

                # 只拷贝尚未缓存的token（仅完整的block）
                start_token = seq.num_cached_tokens
                start_block_idx = start_token // block_size
                token_offset = start_token
                
                # gpu_kv_cache形状: [2, num_layers, num_blocks, block_size, num_kv_heads, head_dim]
                # cpu_kv_cache形状: [2, num_layers, seq_len, num_kv_heads, head_dim]
                
                # 跳过已完全缓存的前导block
                for block_id in seq.block_table[start_block_idx:]:
                    # 检查是否需要取消操作
                    if cancel_event.is_set():
                        break

                    # 计算剩余需要缓存的token数量
                    remaining = seq.text_token_len - token_offset
                    if remaining <= 0:
                        break

                    # 计算当前block需要拷贝的token数量
                    block_tokens = remaining if remaining < block_size else block_size

                    # 对每一对(kv_idx, layer_idx)执行拷贝
                    for kv_idx in range(2):  # Key和Value
                        for layer_idx in range(num_layers):
                            # 目标：GPU缓存中的对应位置
                            dst = self.gpu_kv_cache[
                                kv_idx, layer_idx, block_id, :block_tokens
                            ]
                            # 源：CPU缓存中对应的token范围
                            src = cpu_kv_cache[
                                kv_idx,
                                layer_idx,
                                token_offset : token_offset + block_tokens,
                            ]
                            # 执行一次拷贝：CPU -> GPU（如果src是pinned memory则non_blocking）
                            dst.copy_(src, non_blocking=True)
                            any_copied = True

                    # 更新已缓存的token数量
                    token_offset += block_tokens
                    seq.num_cached_tokens = token_offset

        # 如果执行了拷贝操作，记录事件并返回
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
        """
        检查序列是否已被索引
        
        参数:
            seq: 要检查的序列
            
        返回:
            bool: 如果序列有text_id且在索引中，返回True；否则返回False
        """
        return seq.text_id and seq.text_id in self.kv_cache_index

    def persistence(self):
        """
        将KV缓存索引持久化到磁盘
        
        只有当索引被修改过（dirty=True）时才会执行保存操作
        保存后会重置dirty标志
        """
        print("[persistence]")
        if self.dirty:
            path = f"{self.save_dir}{self.index_name}"
            torch.save(self.kv_cache_index, path)
        self.dirty = False


# 以下是使用Triton实现KV缓存加载的尝试代码（目前未使用）
# Triton只能加载GPU内存，因此对于CPU到GPU的传输目前无法使用
# 保留此代码作为未来优化的参考

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
