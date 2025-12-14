import time

from nanovllm import LLM, SamplingParams
from nanovllm.speculative_prefill import enable_prefill_spec


def measure_ttft(llm, prompts, sampling_params):
    start = time.perf_counter()
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    ttft = time.perf_counter() - start
    return outputs, ttft


def main():
    base_model = "Qwen/Qwen3-0.6B-Instruct"
    spec_model = "Qwen/Qwen3-0.6B-Instruct"
    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
    prompts = ["简要说明 speculative prefill 相比直接推理在 TTFT 上的收益。"]

    # Baseline without speculative prefill
    llm_baseline = LLM(base_model, enforce_eager=True, tensor_parallel_size=1)
    base_outputs, base_ttft = measure_ttft(llm_baseline, prompts, sampling_params)

    # Enable speculative prefill then construct LLM to activate the patch
    enable_prefill_spec(spec_model)
    llm_spec = LLM(base_model, enforce_eager=True, tensor_parallel_size=1)
    spec_outputs, spec_ttft = measure_ttft(llm_spec, prompts, sampling_params)

    print("Baseline output:", base_outputs[0]["text"])
    print("Speculative prefill output:", spec_outputs[0]["text"])
    print(f"Baseline TTFT: {base_ttft:.3f}s, Speculative TTFT: {spec_ttft:.3f}s, Delta: {base_ttft - spec_ttft:.3f}s")


if __name__ == "__main__":
    main()
