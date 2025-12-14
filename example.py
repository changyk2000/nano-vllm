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
    path = os.path.expanduser("/data/zwt/model/models/Qwen/Qwen3-8B/")
    llm = LLM(path, enforce_eager=False, tensor_parallel_size=1)
    max_output_len = 1
    num_input_lines = 1000
    sampling_params = SamplingParams(temperature=1, max_tokens=max_output_len)
    dataset = ImdbDataset()

    ###################################################################
    # 第一阶段：构建KV缓存索引或预热
    # 如果索引文件不存在，需要处理所有样本来构建索引
    # 如果索引已存在，只需少量样本预热
    num_warmup = 3
    if not llm.kv_cache_index.indexed:
        num_warmup = num_input_lines

    print(colored("\nBuild index / Warm up", "yellow"))
    base_prompt = " "
    # base_prompt = f'Given the above film review, answer whether the sentiment is "positive" or "negative". Respond ONLY with "positive" or "negative", in all lower case.\n'
    samples, tast_str_len = dataset.sample(base_prompt, num_warmup)
    sampling_params.task_str_len = tast_str_len

    # 使用性能分析器记录索引构建过程
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        profile_memory=True,
        with_stack=False,
    ) as prof:
        start = time()
        # 第一次生成：构建KV缓存索引
        # use_index=True: 启用KV缓存索引功能
        # pruning=True: 启用token剪枝，只保留重要的token
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
    # 第二阶段：使用已构建的KV缓存索引执行新任务
    # 这次不需要重新计算整个文本的KV值，直接从索引中加载
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
        # 第二次生成：复用KV缓存索引
        # use_index=True: 从索引中加载已缓存的KV值
        # pruning=False: 不再需要剪枝，因为已经在第一次处理时完成
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


if __name__ == "__main__":
    main()
