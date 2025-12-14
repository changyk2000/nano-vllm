import math
from typing import List, Tuple

import torch
from transformers import AutoModelForCausalLM

from nanovllm.speculative_prefill.config import SpeculativePrefillConfig


class SpeculativePrefiller:
    def __init__(self, config: SpeculativePrefillConfig):
        self.config = config
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        self.model = AutoModelForCausalLM.from_pretrained(
            config.spec_model,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.model.eval()

    @torch.inference_mode()
    def compress_prompt(self, token_ids: List[int]) -> Tuple[List[int], List[int], int]:
        if len(token_ids) <= 1:
            positions = list(range(len(token_ids)))
            return token_ids, positions, len(token_ids)

        device = next(self.model.parameters()).device
        input_ids = torch.tensor([token_ids], device=device)
        try:
            outputs = self.model(input_ids=input_ids, output_attentions=True, use_cache=True)
            attentions = outputs.attentions
            if not attentions:
                raise RuntimeError("attention not returned")
            scores = torch.stack([layer[0, :, -1, :] for layer in attentions], dim=0).mean(dim=(0, 1))

            past_key_values = outputs.past_key_values
            logits = outputs.logits[:, -1:]
            for _ in range(max(0, self.config.look_ahead_cnt - 1)):
                next_token = torch.argmax(logits, dim=-1)
                outputs = self.model(
                    input_ids=next_token,
                    past_key_values=past_key_values,
                    use_cache=True,
                    output_attentions=True,
                )
                past_key_values = outputs.past_key_values
                logits = outputs.logits
                step_scores = torch.stack(
                    [layer[0, :, -1, :scores.size(0)] for layer in outputs.attentions],
                    dim=0,
                ).mean(dim=(0, 1))
                scores = torch.maximum(scores, step_scores)
        except Exception:
            positions = list(range(len(token_ids)))
            return token_ids, positions, len(token_ids)

        scores[-1] = scores.max()
        keep = max(1, int(math.ceil(len(token_ids) * self.config.keep_percentage)))
        keep = min(keep, len(token_ids))
        indices = torch.topk(scores, k=keep).indices.sort()[0].tolist()
        kept_tokens = [token_ids[i] for i in indices]
        return kept_tokens, indices, len(token_ids)
