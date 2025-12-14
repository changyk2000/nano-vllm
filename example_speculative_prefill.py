from nanovllm import LLM, SamplingParams
from nanovllm.speculative_prefill import enable_prefill_spec


def main():
    # Enable speculative prefill before constructing the LLM instance.
    # Replace the model names with local paths or HF identifiers that you have access to.
    enable_prefill_spec(
        spec_model="meta-llama/Llama-3.2-1B-Instruct",
        spec_config_path=None,  # Optional YAML config that follows the upstream patch format.
    )

    llm = LLM(
        "/YOUR/BASE/MODEL/PATH",
        enforce_eager=True,
        tensor_parallel_size=1,
    )

    sampling_params = SamplingParams(temperature=0.6, max_tokens=64)
    prompts = ["Summarize the benefits of speculative prefill in one sentence."]
    outputs = llm.generate(prompts, sampling_params)
    print(outputs[0]["text"])


if __name__ == "__main__":
    main()
