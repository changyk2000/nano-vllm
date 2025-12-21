import pickle
from concurrent.futures import ThreadPoolExecutor
import os
import random
from termcolor import colored
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import numpy as np

from torch.profiler import record_function

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model


def set_all_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event
        # Dedicated executor to offload compute_logits to a background thread
        self._logits_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="logits")

        self.d2h_stream = torch.cuda.Stream()
        self._host_tokens = None
        # Runtime flag (can be mutated externally by LLMEngine)
        self.pruning_enabled = False
        self.sparsity = 0.9

        # Check if process group is already initialized and destroy it first
        # This can happen when creating multiple LLM instances in the same process
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass  # Already destroyed or not properly initialized
        dist.init_process_group("nccl", "tcp://localhost:2334", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device("cuda")
        self.model = Qwen3ForCausalLM(hf_config)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        # self.warmup_model_with_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

        set_all_seeds()

    def exit(self):
        # Gracefully shutdown background executor
        if self._logits_executor is not None:
            self._logits_executor.shutdown(wait=True, cancel_futures=False)
            self._logits_executor = None
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        # Only destroy process group if it's still initialized
        try:
            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass  # Already destroyed or not properly initialized

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    # def warmup_model_with_cache(self):
    #     block_size = self.config.kvcache_block_size
    #     # 1. prefill warmup
    #     seq = Sequence([0] * block_size)
    #     # fake block table
    #     seq.block_table.append(0)
    #     self.run([seq], True)
    #
    #     # 2. prefix cache warmup
    #     seq = Sequence([0] * block_size * 2)
    #     seq.num_cached_tokens = block_size
    #     # fake block table
    #     seq.block_table.extend([0, 1])
    #     self.run([seq], True)

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size # type: ignore
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * hf_config.head_dim * hf_config.torch_dtype.itemsize # type: ignore
        config.num_kvcache_blocks = int(total * config.gpu_memory_utilization - used - peak + current) // block_bytes # type: ignore
        assert config.num_kvcache_blocks > 0 # type: ignore
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, hf_config.head_dim) # type: ignore
        print(colored(f"num blocks: {config.num_kvcache_blocks}, total: {config.num_kvcache_blocks * self.block_size}", "magenta"))
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens + seq.pruning_len, seqlen + seq.pruning_len)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup
                continue

            last_cached_block_num_tokens = seq.num_cached_tokens - seq.num_cached_blocks * self.block_size
            offset = seq.block_table[seq.num_cached_blocks] * self.block_size
            start = offset + last_cached_block_num_tokens

            if seq.num_cached_blocks != seq.num_blocks - 1:
                end = offset + self.block_size
            else:
                end = offset + seq.last_block_num_tokens

            slot_mapping.extend(list(range(start, end)))

            for i in range(seq.num_cached_blocks + 1, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens 
                slot_mapping.extend(list(range(start, end)))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        # Single context set call including pruning flags (indices discovered inside attention later)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, pruning_enabled=self.pruning_enabled, sparsity=self.sparsity)
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(seq.block_table[-1] * self.block_size + seq.last_block_num_tokens  - 1)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables)
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            # Forward pass on current thread
            hidden = self.model(input_ids, positions)
            # Ensure hidden states are ready before handing over to another thread
            torch.cuda.current_stream().synchronize()
            return self._compute_logits_in_thread(hidden)
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            # Ensure graph outputs are ready before offloading to another thread
            torch.cuda.current_stream().synchronize()
            return self._compute_logits_in_thread(graph_vars["outputs"][:bs])

    @torch.inference_mode()
    def _compute_logits_in_thread(self, hidden: torch.Tensor) -> torch.Tensor:
        """Run model.compute_logits in a dedicated background thread and return logits.

        Note: We synchronize the producing stream before submitting to avoid cross-thread
        stream dependency issues. This keeps correctness while honoring the request
        to execute compute_logits on a separate thread.
        """
        if self._logits_executor is None:
            # Fallback to inline execution if executor is unavailable
            return self.model.compute_logits(hidden)

        @torch.inference_mode()
        def _worker(h: torch.Tensor) -> torch.Tensor:
            # Ensure correct CUDA device is set in this thread
            return self.model.compute_logits(h)

        future = self._logits_executor.submit(_worker, hidden)
        return future.result()

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int] | None:
        input_ids, positions = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        with record_function("forward"):
            logits = self.run_model(input_ids, positions, is_prefill)
            if self.rank == 0:
                tokens = self.sampler(logits, temperatures)  # GPU 张量
                if self._host_tokens is None or self._host_tokens.numel() != tokens.numel():
                    self._host_tokens = torch.empty_like(tokens, device="cpu", pin_memory=True)
                compute_end = torch.cuda.Event()
                torch.cuda.current_stream().record_event(compute_end)  # type: ignore
                with torch.cuda.stream(self.d2h_stream):  # type: ignore
                    self.d2h_stream.wait_event(compute_end)  # type: ignore
                    self._host_tokens.copy_(tokens, non_blocking=True)  # Device -> Pinned
                self.d2h_stream.synchronize()
                token_ids = self._host_tokens.tolist()
            else:
                token_ids = None
        # Capture pruning indices (if any) from context before resetting
        ctx = get_context()
        if is_prefill and ctx.pruning_enabled and ctx.pruned_local_indices is not None:
            # Assign per-sequence pruning indices (local prompt positions)
            for seq, local_idx in zip(seqs, ctx.pruned_local_indices):
                seq.pruning_indices = local_idx.cpu().tolist()  # type: ignore[attr-defined]
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
