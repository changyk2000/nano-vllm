from copy import copy
from enum import Enum, auto
from itertools import count

from nanovllm.sampling_params import SamplingParams


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()


class Sequence:
    block_size = 256
    counter = count()

    def __init__(
        self,
        token_ids: list[int],
        sampling_params=SamplingParams(),
        *,
        position_ids: list[int] | None = None,
        next_position: int | None = None,
    ):
        self.seq_id = next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.token_ids = copy(token_ids)
        self.last_token = token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        if position_ids is not None:
            assert len(position_ids) == self.num_tokens
        self.position_ids = copy(position_ids) if position_ids is not None else list(range(self.num_tokens))
        self.next_position = next_position if next_position is not None else self.num_tokens

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
        self.position_ids.append(self.next_position)
        self.next_position += 1

    def __getstate__(self):
        return dict(
            num_tokens=self.num_tokens,
            num_prompt_tokens=self.num_prompt_tokens,
            num_cached_tokens=self.num_cached_tokens,
            block_table=self.block_table,
            token_ids=self.token_ids,
            position_ids=self.position_ids,
            next_position=self.next_position,
        )

    def __setstate__(self, state):
        self.num_tokens = state["num_tokens"]
        self.num_prompt_tokens = state["num_prompt_tokens"]
        self.num_cached_tokens = state["num_cached_tokens"]
        self.block_table = state["block_table"]
        self.token_ids = state["token_ids"]
        self.position_ids = state["position_ids"]
        self.next_position = state["next_position"]
        self.last_token = self.token_ids[-1]
