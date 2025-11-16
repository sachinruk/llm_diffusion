import random
from typing import Any

import datasets
import torch
import transformers
from loguru import logger

from src import config


def load_dataset(
    hyper_parameters: config.HyperParameters,
) -> tuple[datasets.Dataset, datasets.Dataset]:
    raw_dataset: datasets.Dataset = datasets.load_dataset(
        hyper_parameters.dataset,
        split="train",  # [:1%] % for demo.  drop the slice for real training
        cache_dir="./data",
        download_mode=datasets.DownloadMode.REUSE_DATASET_IF_EXISTS,
    )
    if hyper_parameters.debug:
        raw_dataset = raw_dataset.select(range(100))
        logger.info(f"Debug mode: {raw_dataset.num_rows} rows")
    else:
        logger.info(f"Training mode: {raw_dataset.num_rows} rows")
    raw_dataset_dict = raw_dataset.train_test_split(
        test_size=hyper_parameters.test_size, seed=hyper_parameters.seed
    )
    return raw_dataset_dict["train"], raw_dataset_dict["test"]


def mask_input_ids(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mask_probability: float,
    mask_token_id: int,
    response_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mask input IDs.

    Args:
        input_ids: The input IDs to mask.
        attention_mask: The attention mask.
        mask_probability: The probability of masking an input ID.
        mask_token_id: The ID of the mask token.
        response_mask: The mask to apply to the response (optional). Must be boolean.

    Returns:
        The masked input IDs.
    """
    mask_indices = torch.bernoulli(
        torch.full(size=input_ids.shape, fill_value=mask_probability, device=input_ids.device)
    ).bool()
    mask_indices = mask_indices & attention_mask.bool()
    if response_mask is not None:
        mask_indices: torch.Tensor = mask_indices & response_mask
    input_ids[mask_indices] = mask_token_id
    return input_ids


def get_prefix_mask(input_ids: torch.Tensor, separator_token_id: int) -> torch.Tensor:
    """Mask labels before the last separator token.

    Args:
        labels: The labels to mask.
        separator_token_id: The ID of the separator token.
        ignore_index: The index to use for the ignored labels.

    Returns:
        The masked labels.
    """
    sep_mask = input_ids == separator_token_id  # (B, L) booleans
    s = sep_mask.cumsum(dim=1)  # running count of seps
    total = s[:, -1:]  # total seps per row (B, 1)
    last_sep_onehot = sep_mask & (s == total)  # 1 only at the last sep (or all 0 if none)

    # Build a mask of positions <= last separator (inclusive):
    # reverse -> cumsum -> reverse gives ones from start up to that last-sep index
    prefix_mask = last_sep_onehot.flip(dims=[1]).cumsum(dim=1).flip(dims=[1]).bool()
    prefix_mask[(total == 0).squeeze()] = True
    return prefix_mask


def tokenize_text(
    examples: list[dict[str, Any]],
    tokenizer: transformers.PreTrainedTokenizer,
    max_length: int = 1024,
) -> transformers.BatchEncoding:
    string_examples: list[str] = tokenizer.apply_chat_template(
        [example["messages"] for example in examples],
        tokenize=False,
        add_generation_prompt=False,
    )

    return tokenizer(
        string_examples,
        padding=True,
        max_length=max_length,
        return_tensors="pt",
        truncation=True,
    )


class CollateFn:
    def __init__(
        self,
        tokenizer: transformers.PreTrainedTokenizer,
        max_length: int = 1024,
        min_mask_probability: float = 0.1,
        max_mask_probability: float = 0.95,
    ):
        self.tokenizer = tokenizer
        self.mask_token_id: int = tokenizer.convert_tokens_to_ids(config.MASK_TOKEN)
        self.sep_token_id: int = tokenizer.convert_tokens_to_ids(config.IM_START_TOKEN)
        self.ignore_index = -100
        self.max_length = max_length
        self.min_mask_probability = min_mask_probability
        self.max_mask_probability = max_mask_probability


class PretrainingCollateFn(CollateFn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, examples: list[dict[str, Any]]) -> transformers.BatchEncoding:
        encoded_batch = tokenize_text(examples, self.tokenizer, self.max_length)
        input_ids: torch.Tensor = encoded_batch["input_ids"]
        attention_mask: torch.Tensor = encoded_batch["attention_mask"]

        labels: torch.Tensor = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index

        input_ids = mask_input_ids(
            input_ids,
            attention_mask,
            mask_probability=random.uniform(self.min_mask_probability, self.max_mask_probability),
            mask_token_id=self.mask_token_id,
        )
        encoded_batch["labels"] = labels

        return encoded_batch


class SFTCollateFn(CollateFn):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, examples: list[dict[str, Any]]) -> transformers.BatchEncoding:
        encoded_batch = tokenize_text(examples, self.tokenizer, self.max_length)
        input_ids: torch.Tensor = encoded_batch["input_ids"]
        attention_mask: torch.Tensor = encoded_batch["attention_mask"]

        prefix_mask = get_prefix_mask(input_ids, self.sep_token_id)
        labels: torch.Tensor = input_ids.clone()
        labels[attention_mask == 0] = self.ignore_index
        labels[prefix_mask] = self.ignore_index

        input_ids = mask_input_ids(
            input_ids,
            attention_mask,
            mask_probability=random.uniform(self.min_mask_probability, self.max_mask_probability),
            mask_token_id=self.mask_token_id,
            response_mask=~prefix_mask,
        )

        encoded_batch["labels"] = labels
        return encoded_batch
