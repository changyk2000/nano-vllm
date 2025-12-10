import atexit
import threading
from dataclasses import fields
from queue import Empty, Queue
from time import perf_counter, sleep, time

import torch
import torch.multiprocessing as mp
from termcolor import colored
from torch.profiler import record_function
from tqdm.auto import tqdm
from transformers import AutoTokenizer

from nanovllm.config import Config
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.sequence import Sequence
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.kv_cache_index import KVCacheIndex


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        assert config.tensor_parallel_size == 1  # not supported for now
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        # 初始化KV缓存索引，传入GPU上的KV缓存引用
        # 用于管理长文本的KV缓存持久化和复用
        self.kv_cache_index = KVCacheIndex(self.model_runner.kv_cache)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)

        # Prefetch machinery
        self._prefetch_thread: threading.Thread
        self._prefetch_queue: Queue
        self._cancel_prefetch = threading.Event()

        # Store machinery
        self._store_thread: threading.Thread
        self._store_queue: Queue

        # Stats
        self._stats = {
            "total_xfer_ms": 0.0,
            "total_compute_ms": 0.0,
            "num_batches": 0,
        }
        self.last_run_stats: dict[str, float] | None = None
        atexit.register(self.exit)
        self.use_index = False

    def exit(self):
        """退出时的清理工作"""
        # 如果使用了索引，在退出前持久化KV缓存索引到磁盘
        if self.use_index:
            self.kv_cache_index.persistence()

        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def add_request(
        self, prompt: str | list[int] | tuple[int, str], sampling_params: SamplingParams, use_index
    ):
        """
        添加推理请求到调度器
        
        参数:
            prompt: 输入提示，可以是：
                - str: 原始文本
                - list[int]: token ID列表
                - tuple[int, str]: (text_id, 文本内容)，用于KV缓存索引
            sampling_params: 采样参数
            use_index: 是否使用KV缓存索引
        """
        text_token_ids = []
        pruning_len = 0
        if isinstance(prompt, str):
            # 情况1：原始文本，直接编码
            prompt = self.tokenizer.encode(prompt)
        elif isinstance(prompt, tuple):
            # 情况2：带有text_id的元组格式
            if use_index:
                text_id = prompt[0]
                # 尝试从KV缓存索引中获取已缓存的数据
                item = self.kv_cache_index.kv_cache_index.get(text_id)
                if isinstance(item, dict):
                    # 如果找到了缓存，使用剪枝后的token列表
                    # 这样可以避免重新编码文本，节省时间
                    text_token_ids = item.get("text_tokens_pruned")
                    # print(f"[kept text]: { self.tokenizer.decode(text_token_ids) }")
                    pruning_len = item.get("pruning_len")  # type: ignore
                else:
                    # 如果没有缓存，需要重新编码文本部分
                    text_token_ids = self.tokenizer.encode(prompt[1][:len(prompt[1])-sampling_params.task_str_len])
                # 编码任务提示部分
                task_token_ids = self.tokenizer.encode(prompt[1][-sampling_params.task_str_len:])
                # 组合文本token和任务token
                prompt = (text_id, text_token_ids + task_token_ids)
            else:
                # 不使用索引时，直接编码整个文本
                prompt = (prompt[0], self.tokenizer.encode(prompt[1]))

        # print(f"input len: {len(prompt[1])}") # type: ignore
        # 创建序列对象并添加到调度器
        seq = Sequence(prompt, len(text_token_ids), pruning_len, sampling_params)  # type: ignore
        self.scheduler.add(seq)

    def _start_prefetcher(self):
        """
        启动预取线程，用于异步加载KV缓存
        
        该方法创建一个后台线程，负责：
        1. 调度序列并确定需要加载的KV缓存
        2. 异步地从CPU加载KV缓存到GPU
        3. 将准备好的批次放入队列供主线程处理
        
        这种设计可以实现KV缓存加载与模型推理的并行执行
        """
        self._prefetch_queue: Queue = Queue(maxsize=8)

        def _prefetch_loop():
            """预取线程的主循环"""
            # 创建专用的CUDA流用于KV缓存传输
            prefetch_stream = torch.cuda.Stream()
            while True:
                # 尽可能多地调度序列以填充GPU blocks
                try:
                    seqs, is_prefill = self.scheduler.schedule()
                except AssertionError:
                    # 当前没有可调度的序列
                    if self.scheduler.is_finished():
                        break
                    sleep(0.001)
                    continue

                # 如果序列已被索引，启动H2D（CPU到GPU）KV缓存传输
                transfer_event = None
                xfer_ms = 0.0
                with record_function("get kv index"):
                    ret = self.kv_cache_index.get_kv_cache(
                        seqs,
                        stream=prefetch_stream,  # type: ignore
                        return_timing=True,
                        cancel_event=self._cancel_prefetch,
                    )

                # 如果取消标志被设置，清除它
                if self._cancel_prefetch.is_set():
                    self._cancel_prefetch.clear()

                # 如果有KV缓存传输，等待传输完成并记录时间
                if ret is not None:
                    if isinstance(ret, tuple):
                        transfer_event, start_event = ret
                    else:
                        transfer_event, start_event = ret, None
                    transfer_event.synchronize()
                    if start_event is not None:
                        xfer_ms = start_event.elapsed_time(transfer_event)
                
                # 将准备好的批次放入队列供计算使用
                self._prefetch_queue.put((seqs, is_prefill, xfer_ms))

            print(colored("prefetch thread quit!", "red"))

        # 启动预取线程
        self._prefetch_thread = threading.Thread(
            target=_prefetch_loop, name="kv-prefetch", daemon=True
        )
        self._prefetch_thread.start()

    def _start_storer(self):
        """
        启动存储线程，用于异步保存KV缓存
        
        该方法创建一个后台线程，负责：
        1. 接收需要保存的序列
        2. 异步地将GPU上的KV缓存转移到CPU内存
        3. 在转移完成后释放GPU blocks
        
        这种设计可以实现KV缓存保存与模型推理的并行执行
        """
        # 初始化就绪队列和传输流
        self._store_queue = Queue(maxsize=-1)

        def _store_loop():
            """存储线程的主循环"""
            # 创建专用的CUDA流用于KV缓存传输
            store_stream = torch.cuda.Stream()
            while True:
                # 从队列获取需要存储的序列
                transfer_event = None
                seqs = self._store_queue.get()
                # 空列表是退出信号
                if len(seqs) == 0:
                    break

                # 执行KV缓存存储（D2H: GPU到CPU传输）
                with record_function("store kv index"):
                    ret = self.kv_cache_index.store_kv_cache(
                        seqs, stream=store_stream, return_timing=False  # type: ignore
                    )

                # 如果有传输事件，等待传输完成
                if ret is not None:
                    if isinstance(ret, tuple):
                        transfer_event, start_event = ret
                    else:
                        transfer_event, start_event = ret, None
                    transfer_event.synchronize()

                    if start_event is not None:
                        print(
                            f"store kv cache: {start_event.elapsed_time(transfer_event)}"
                        )

                # 在这里释放blocks
                # 只有在KV缓存转移到CPU后才能安全释放GPU内存
                for seq in seqs:
                    seq.lock_block = False
                    self.scheduler.block_manager.deallocate(seq)

            print(colored("store thread quit!", "red"))

        # 启动存储线程
        self._store_thread = threading.Thread(
            target=_store_loop, name="kv-store", daemon=True
        )
        self._store_thread.start()

    def step(self, use_index):
        """
        执行一个推理步骤
        
        参数:
            use_index: 是否使用KV缓存索引
            
        返回:
            outputs: 完成的序列输出列表 [(seq_id, token_ids), ...]
            num_tokens: 处理的token数量（正数表示prefill，负数表示decode）
        """
        xfer_ms = 0.0
        if use_index:
            # 从预取队列中获取准备好的批次（阻塞直到可用）
            try:
                # item = self._prefetch_queue.get_nowait()
                item = self._prefetch_queue.get()
            except Empty:
                # 如果队列为空，设置取消标志并重试
                self._cancel_prefetch.set()
                item = self._prefetch_queue.get()

            # 解包批次数据（支持向后兼容）
            seqs, is_prefill, xfer_ms = item
            if not seqs and self.scheduler.is_finished():
                return [], 0
            print(colored(f"schedule {len(seqs)} seq", "magenta"))
        else:
            # 不使用索引时，直接调度序列
            seqs, is_prefill = self.scheduler.schedule()
            print(colored(f"schedule {len(seqs)} seq", "magenta"))

        # 执行模型推理
        start = time()
        with record_function("run model"):
            token_ids = self.model_runner.call("run", seqs, is_prefill)
        end = time()
        compute_ms = (end - start) * 1000

        if use_index:
            # 使用索引时，锁定序列的blocks并将其加入存储队列
            # 锁定是为了防止在KV缓存转移到CPU之前被释放
            for seq in seqs:
                seq.lock_block = True
            assert len(seqs) > 0
            # 将序列放入存储队列，由存储线程异步处理
            self._store_queue.put_nowait(seqs)  # type: ignore

        # 后处理：更新序列状态
        self.scheduler.postprocess(seqs, token_ids)
        
        # 统计和打印信息
        if use_index:
            self._stats["total_xfer_ms"] += xfer_ms
            self._stats["total_compute_ms"] += compute_ms
            self._stats["num_batches"] += 1
            print(
                colored(f"H2D: {xfer_ms:.2f} ms | Compute: {compute_ms:.2f} ms", "cyan")
            )
        outputs = [
            (seq.seq_id, seq.completion_token_ids) for seq in seqs if seq.is_finished
        ]
        num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
        return outputs, num_tokens

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]] | list[tuple[int, str]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
        use_index: bool = False,
        pruning: bool = False,
        sparsity: float = 0.9,
    ) -> list[dict]:
        self.model_runner.sparsity = sparsity
        init_start = time()
        self.last_run_stats = None
        self.use_index = self.use_index or use_index
        # Toggle pruning feature for this generation session
        # Propagate to model runner so it can prepare the runtime context.
        self.model_runner.pruning_enabled = pruning  # type: ignore[attr-defined]
        if use_tqdm:
            pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp, use_index)
        if use_index:
            # Start prefetch worker once requests are queued
            self._start_prefetcher()
            self._start_storer()
        outputs = {}
        prefill_throughput = decode_throughput = 0.0
        end = time()
        print(f"init: {end - init_start}")
        run_start = perf_counter()
        while not self.is_finished() or (use_index and not self._prefetch_queue.empty()):
            t = perf_counter()
            output, num_tokens = self.step(use_index)
            if use_tqdm:
                if num_tokens > 0:
                    prefill_throughput = num_tokens / (perf_counter() - t)
                else:
                    decode_throughput = -num_tokens / (perf_counter() - t)
                pbar.set_postfix( # type: ignore
                    {  # type: ignore
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    }
                )
            for seq_id, token_ids in output:
                outputs[seq_id] = token_ids
                if use_tqdm:
                    pbar.update(1)  # type: ignore
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {"text": self.tokenizer.decode(token_ids), "token_ids": token_ids}
            for token_ids in outputs
        ]
        if use_tqdm:
            pbar.close()  # type: ignore
        torch.cuda.synchronize()
        run_duration_ms = (perf_counter() - run_start) * 1000
        self.last_run_stats = {
            "avg_transfer_ms": 0.0,
            "avg_compute_ms": run_duration_ms,
            "num_batches": 0.0,
            "use_index": use_index,
            "runtime_ms": run_duration_ms,
        }
        if use_index:
            total_xfer = self._stats["total_xfer_ms"]
            total_comp = self._stats["total_compute_ms"]
            nb = max(self._stats["num_batches"], 1)
            avg_xfer = total_xfer / nb if nb else 0.0
            avg_comp = total_comp / nb if nb else 0.0
            print(
                colored(
                    f"\nTiming summary (per batch): H2D avg {avg_xfer:.2f} ms | Compute avg {avg_comp:.2f} ms | batches {nb}",
                    "green",
                )
            )
            self.last_run_stats = {
                "avg_transfer_ms": avg_xfer,
                "avg_compute_ms": avg_comp,
                "num_batches": float(nb),
                "use_index": True,
                "runtime_ms": run_duration_ms,
            }
            self._stats = {
                "total_xfer_ms": 0.0,
                "total_compute_ms": 0.0,
                "num_batches": 0,
            }
            # join background threads
            if self._prefetch_thread is not None:
                self._prefetch_thread.join()

            if self._store_thread is not None:
                self._store_queue.put_nowait([])  # type: ignore
                self._store_thread.join()
        return outputs
