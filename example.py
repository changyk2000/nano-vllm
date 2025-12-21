import os
from time import time

import pandas as pd
from termcolor import colored
from torch.profiler import ProfilerActivity, profile

from nanovllm import LLM, SamplingParams


class ImdbDataset:
    """
    Simplified implementation of the Sonnet dataset.  Loads poem lines from a
    text file and generates sample requests.  Default values here copied from
    `benchmark_serving.py` for the sonnet dataset.
    """

    DEFAULT_OUTPUT_LEN = 150

    def __init__(
        self,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.load_data()

    def load_data(self) -> None:
        self.data = pd.read_csv("./data/imdb.csv")

    def sample(
        self,
        base_prompt,
        num_input_lines=1000,
    ) -> tuple[list[tuple[int, str]], int]:
        chose_lines = self.data["review"][:num_input_lines]
        lines_per_prompt = 1
        assert lines_per_prompt == 1
        duplicate = 1
        samples = []
        num_requests = int(num_input_lines / lines_per_prompt)
        task_str_len = len(base_prompt)

        for i in range(num_requests):
            texts = "\n".join(
                chose_lines[i * lines_per_prompt : (i + 1) * lines_per_prompt]
            )
            for _ in range(duplicate):
                # base_prompt = f'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
                # base_prompt = f'Given the above film review, answer whether the film is suitable for kids. Respond ONLY with "yes" or "no", in all lower case.\n'
                # base_prompt = f"Given the above film review, answer whether it contains names. Respond ONLY with \"yes\" or \"no\", in all lower case.\n"
                # base_prompt = f"Given the above film review, answer whether it contains violent elements. Respond ONLY with \"yes\" or \"no\", in all lower case.\n"
                prompt = f"{texts}\n{base_prompt}"
                samples.append((i, prompt))
        return samples, task_str_len


def main():
    # Init
    path = os.path.expanduser("/data/zhangyuyun/model/models/Qwen/Qwen3-0.6B")
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
    max_output_len = 1
    num_input_lines = 1000
    sampling_params = SamplingParams(temperature=1, max_tokens=max_output_len)
    dataset = ImdbDataset()

    ###################################################################
    num_warmup = 3
    if not llm.kv_cache_index.indexed:
        num_warmup = num_input_lines

    print(colored("\nBuild index / Warm up", "yellow"))
    base_prompt = " "
    # base_prompt = f'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
    samples, tast_str_len = dataset.sample(base_prompt, num_warmup)
    sampling_params.task_str_len = tast_str_len

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time()
        outputs = llm.generate(
            samples, sampling_params, use_index=True, use_tqdm=False, pruning=True
        )
        end = time()
    prof.export_chrome_trace("trace_task1.json")

    print(colored(f"Build index time: {(end - start):.4f} s", "blue"))

    # remove prefix cache
    # llm.scheduler.block_manager.reset()

    ###################################################################
    # print(colored("Task1: suitable for kids", "yellow"))
    #
    # base_prompt = f'Given the above film review, answer whether the film is suitable for kids. Respond ONLY with "yes" or "no", in all lower case.\n'
    # samples, base_token_len = dataset.sample(base_prompt, 20)
    # sampling_params.base_token_len = base_token_len
    #
    # # The first task will generate kv cache index for texts
    # # with profile(
    # #     activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    # #     profile_memory=True,
    # #     with_stack=True,
    # # ) as prof:
    # start = time()
    # outputs = llm.generate(samples, sampling_params, use_index=True, use_tqdm=False)
    # end = time()
    # # prof.export_chrome_trace("trace_task1.json")
    #
    # print(colored(f"Total generate time: {( end - start ):.4f} s", "blue"))
    #
    # generated = [output["text"] for output in outputs]
    # # print(f"{generated[:10]}")

    ###################################################################
    print(colored("\nTask2: sentiment", "yellow"))
    base_prompt = f'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
    samples, tast_str_len = dataset.sample(base_prompt, num_input_lines)
    sampling_params.task_str_len = tast_str_len

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time()
        outputs = llm.generate(
            samples, sampling_params, use_index=True, use_tqdm=False
        )
        end = time()
    prof.export_chrome_trace("trace_task2.json")

    print(colored(f"Total generate time: {( end - start ):.4f} s", "blue"))
    print(f"output: {len(outputs)}")

    generated = [output["text"] for output in outputs]
    print(f"{generated[:10]}")

    data = pd.read_csv("./data/imdb.csv").head(len(outputs))
    correct_predictions = (data["sentiment"] == generated).sum()
    accuracy = correct_predictions / len(outputs)
    print(f"Task 2 Accuracy:{accuracy}\n")

    # print(data[data["suitable"] != generated]["review"])
    # TODO: restrict output token ids
    print("--- Checking for Incorrect and Invalid Results ---")
    mismatched_count = 0
    for i, (gen_text, true_label) in enumerate(zip(generated, data["sentiment"])):
        if gen_text.lower() not in ["positive", "negative"]:
            print(
                f"Index {i}: Invalid output. Generated: '{gen_text}', Expected: '{true_label}'"
            )
            mismatched_count += 1

    if mismatched_count == 0:
        print("No incorrect or invalid results found.")
    print("--- End of Check ---\n")


def benchmark_gpudirect():
    """
    Benchmark comparing KV cache loading time with and without GPUDirect Storage.
    
    This example demonstrates the performance difference between:
    1. Standard CPU->GPU transfer (pinned memory)
    2. GPUDirect Storage (direct SSD->GPU transfer, bypassing CPU)
    
    Note: GPUDirect requires:
    - NVIDIA GPU with GPUDirect Storage support
    - kvikio library installed (`pip install kvikio`)
    - Properly configured GDS drivers
    """
    import torch
    
    print(colored("\n" + "=" * 70, "cyan"))
    print(colored("GPUDirect Storage Benchmark: KV Cache Loading", "cyan"))
    print(colored("=" * 70, "cyan"))
    
    path = os.path.expanduser("/data/zhangyuyun/model/models/Qwen/Qwen3-0.6B")
    max_output_len = 1
    num_input_lines = 100  # Use smaller dataset for benchmark
    dataset = ImdbDataset()
    
    results = {}
    
    # Test 1: Standard transfer (without GPUDirect)
    print(colored("\n[Test 1] Standard CPU->GPU Transfer (without GPUDirect)", "yellow"))
    print("-" * 50)
    
    llm_standard = LLM(path, enforce_eager=False, tensor_parallel_size=1, use_gpudirect=False)
    sampling_params = SamplingParams(temperature=1, max_tokens=max_output_len)
    
    # Warm up / build index
    num_warmup = num_input_lines if not llm_standard.kv_cache_index.indexed else 3
    base_prompt = " "
    samples, tast_str_len = dataset.sample(base_prompt, num_warmup)
    sampling_params.task_str_len = tast_str_len
    
    start = time()
    llm_standard.generate(samples, sampling_params, use_index=True, use_tqdm=False, pruning=True)
    build_time_standard = time() - start
    
    # Test sentiment task
    base_prompt = 'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
    samples, tast_str_len = dataset.sample(base_prompt, num_input_lines)
    sampling_params.task_str_len = tast_str_len
    
    start = time()
    outputs_standard = llm_standard.generate(samples, sampling_params, use_index=True, use_tqdm=False)
    inference_time_standard = time() - start
    
    results["standard"] = {
        "build_time": build_time_standard,
        "inference_time": inference_time_standard,
        "avg_transfer_ms": llm_standard.last_run_stats.get("avg_transfer_ms", 0) if llm_standard.last_run_stats else 0,
    }
    
    print(colored(f"Build index time: {build_time_standard:.4f} s", "blue"))
    print(colored(f"Inference time: {inference_time_standard:.4f} s", "blue"))
    print(colored(f"Avg H2D transfer: {results['standard']['avg_transfer_ms']:.2f} ms", "blue"))
    
    # Clean up
    del llm_standard
    torch.cuda.empty_cache()
    
    # Test 2: GPUDirect transfer
    print(colored("\n[Test 2] GPUDirect Storage Transfer (SSD->GPU)", "yellow"))
    print("-" * 50)
    
    try:
        llm_gpudirect = LLM(path, enforce_eager=False, tensor_parallel_size=1, use_gpudirect=True)
        sampling_params = SamplingParams(temperature=1, max_tokens=max_output_len)
        
        # Warm up / build index
        num_warmup = num_input_lines if not llm_gpudirect.kv_cache_index.indexed else 3
        base_prompt = " "
        samples, tast_str_len = dataset.sample(base_prompt, num_warmup)
        sampling_params.task_str_len = tast_str_len
        
        start = time()
        llm_gpudirect.generate(samples, sampling_params, use_index=True, use_tqdm=False, pruning=True)
        build_time_gpudirect = time() - start
        
        # Test sentiment task
        base_prompt = 'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
        samples, tast_str_len = dataset.sample(base_prompt, num_input_lines)
        sampling_params.task_str_len = tast_str_len
        
        start = time()
        outputs_gpudirect = llm_gpudirect.generate(samples, sampling_params, use_index=True, use_tqdm=False)
        inference_time_gpudirect = time() - start
        
        results["gpudirect"] = {
            "build_time": build_time_gpudirect,
            "inference_time": inference_time_gpudirect,
            "avg_transfer_ms": llm_gpudirect.last_run_stats.get("avg_transfer_ms", 0) if llm_gpudirect.last_run_stats else 0,
        }
        
        print(colored(f"Build index time: {build_time_gpudirect:.4f} s", "blue"))
        print(colored(f"Inference time: {inference_time_gpudirect:.4f} s", "blue"))
        print(colored(f"Avg transfer: {results['gpudirect']['avg_transfer_ms']:.2f} ms", "blue"))
        
        # Clean up
        del llm_gpudirect
        torch.cuda.empty_cache()
        
    except Exception as e:
        print(colored(f"GPUDirect test failed: {e}", "red"))
        print(colored("Make sure kvikio is installed and GDS drivers are configured.", "red"))
        results["gpudirect"] = None
    
    # Print comparison summary
    print(colored("\n" + "=" * 70, "cyan"))
    print(colored("Benchmark Summary", "cyan"))
    print(colored("=" * 70, "cyan"))
    
    print(colored("\n                    Standard    GPUDirect    Speedup", "white"))
    print("-" * 55)
    
    if results.get("gpudirect"):
        build_speedup = results["standard"]["build_time"] / results["gpudirect"]["build_time"] if results["gpudirect"]["build_time"] > 0 else 0
        inference_speedup = results["standard"]["inference_time"] / results["gpudirect"]["inference_time"] if results["gpudirect"]["inference_time"] > 0 else 0
        transfer_speedup = results["standard"]["avg_transfer_ms"] / results["gpudirect"]["avg_transfer_ms"] if results["gpudirect"]["avg_transfer_ms"] > 0 else 0
        
        print(f"Build Index:        {results['standard']['build_time']:8.2f}s   {results['gpudirect']['build_time']:8.2f}s   {build_speedup:5.2f}x")
        print(f"Inference:          {results['standard']['inference_time']:8.2f}s   {results['gpudirect']['inference_time']:8.2f}s   {inference_speedup:5.2f}x")
        print(f"Avg Transfer (ms):  {results['standard']['avg_transfer_ms']:8.2f}    {results['gpudirect']['avg_transfer_ms']:8.2f}    {transfer_speedup:5.2f}x")
    else:
        print(f"Build Index:        {results['standard']['build_time']:8.2f}s   N/A          N/A")
        print(f"Inference:          {results['standard']['inference_time']:8.2f}s   N/A          N/A")
        print(f"Avg Transfer (ms):  {results['standard']['avg_transfer_ms']:8.2f}    N/A          N/A")
        print(colored("\nNote: GPUDirect test was skipped or failed.", "yellow"))
    
    print(colored("\n" + "=" * 70 + "\n", "cyan"))
    
    return results


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "--benchmark-gpudirect":
        # Run GPUDirect benchmark
        benchmark_gpudirect()
    else:
        # Run standard example
        main()
