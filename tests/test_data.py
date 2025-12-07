import random
from unittest.mock import MagicMock

import pytest
import torch

from src import data


@pytest.fixture(autouse=True)
def seed_everything():
    """Seed all random number generators before each test."""
    torch.manual_seed(42)
    random.seed(42)


def test_mask_input_ids():
    input_ids = torch.tensor([[1, 2, 3, 4, 5]])
    attention_mask = torch.tensor([[1, 1, 1, 1, 1]])
    mask_probability = 0.5
    mask_token_id = 99
    response_mask = torch.tensor([[False, False, False, False, False]])
    expected_output = input_ids.clone()
    # no change expected if no response mask is provided
    assert torch.all(
        data.mask_input_ids(
            input_ids, attention_mask, mask_probability, mask_token_id, response_mask
        )
        == expected_output
    )

    mask_probability = 0.99999
    response_mask = torch.tensor([[True, True, True, True, True]])
    expected_output = torch.tensor([[99, 99, 99, 99, 99]])
    assert torch.all(
        data.mask_input_ids(
            input_ids, attention_mask, mask_probability, mask_token_id, response_mask
        )
        == expected_output
    )


def test_get_prefix_mask():
    input_ids = torch.tensor(
        [
            [1, 2, 3, 4, 5],
            [1, 2, 99, 4, 5],
            [1, 2, 99, 99, 5],
            [1, 2, 99, 4, 99],
        ]
    )
    separator_token_id = 99
    expected_output = torch.tensor(
        [
            [True, True, True, True, True],
            [True, True, True, False, False],
            [True, True, True, True, False],
            [True, True, True, True, True],
        ]
    )
    assert torch.all(data.get_prefix_mask(input_ids, separator_token_id) == expected_output)


class MockTokenizer:
    """Mock tokenizer for testing."""

    def convert_tokens_to_ids(self, token: str) -> int:
        """Return mock token IDs."""
        if token == "<|mask|>":
            return 99
        elif token == "<|im_start|>":
            return 100
        return 1

    def apply_chat_template(
        self,
        messages_list: list[list],
        tokenize: bool = False,
        add_generation_prompt: bool = False,
    ) -> list[str]:
        """Mock chat template application."""
        return ["<|im_start|>user\nHello<|im_start|>assistant\nHi there"]

    def __call__(
        self,
        texts: list[str],
        padding: bool = True,
        max_length: int = 1024,
        return_tensors: str = "pt",
        truncation: bool = True,
        padding_side: str = "right",
    ) -> dict[str, torch.Tensor]:
        """Mock tokenization that returns a simple batch."""
        # Simulate: [<im_start>, user_token, user_token, <im_start>, assistant_token, assistant_token]
        # The last <im_start> separates prompt from response
        input_ids = torch.tensor([[100, 1, 2, 100, 3, 4]])
        attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
        return {"input_ids": input_ids, "attention_mask": attention_mask}


def test_sft_collate_fn_full_masking():
    """Test SFTCollateFn with mask probability set to 1.0 for both min and max."""
    mock_tokenizer = MockTokenizer()

    # Create SFTCollateFn with full masking (min=1.0, max=1.0)
    collate_fn = data.SFTCollateFn(
        tokenizer=mock_tokenizer,  # type: ignore
        max_length=1024,
        min_mask_probability=1.0,
        max_mask_probability=1.0,
    )

    # Create sample input
    examples = [{"messages": [{"role": "user", "content": "Hello"}]}]

    # Call the collate function
    batch = collate_fn(examples)

    # Verify output structure
    assert "input_ids" in batch
    assert "attention_mask" in batch
    assert "labels" in batch

    input_ids: torch.Tensor = batch["input_ids"]  # type: ignore
    labels: torch.Tensor = batch["labels"]  # type: ignore

    # Expected behavior with mask_probability=1.0:
    # - All tokens after the last separator (100) should be masked in input_ids
    # - Labels should have prefix masked with ignore_index (-100)
    # Expected input_ids: prefix tokens unchanged, response tokens (positions 4, 5) masked with 99
    expected_input_ids = torch.tensor([[100, 1, 2, 100, 99, 99]])
    assert torch.equal(input_ids, expected_input_ids)

    # Expected labels: prefix masked with -100, response has original tokens
    expected_labels = torch.tensor([[-100, -100, -100, -100, 3, 4]])
    assert torch.equal(labels, expected_labels)


def test_inference_collate_fn():
    """Test InferenceCollateFn returns encoded batch with mask tokens appended."""
    mock_tokenizer = MagicMock()
    mask_token_id = 99
    mock_tokenizer.convert_tokens_to_ids.return_value = mask_token_id
    mock_tokenizer.apply_chat_template.return_value = [
        "<|im_start|>user\nWhat is 2+2?<|im_start|>assistant\n"
    ]

    # First call: tokenize_text for question, Second call: tokenize answers for length
    question_input_ids = torch.tensor([[100, 1, 2, 3, 100, 4]])
    question_attention_mask = torch.tensor([[1, 1, 1, 1, 1, 1]])
    answer_token_ids = [[5, 6, 7]]  # Answer has 3 tokens

    mock_tokenizer.side_effect = [
        # First call: tokenize_text for the question
        {
            "input_ids": question_input_ids,
            "attention_mask": question_attention_mask,
        },
        # Second call: tokenize answers (returns list, not tensor)
        {"input_ids": answer_token_ids},
    ]

    mask_token_buffer = 5
    collate_fn = data.InferenceCollateFn(
        tokenizer=mock_tokenizer,
        max_length=1024,
        mask_token_buffer=mask_token_buffer,
    )

    examples = [
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": "4"},
            ]
        }
    ]

    encoded_batch, actual_answers = collate_fn(examples)

    # Verify output structure
    assert "input_ids" in encoded_batch
    assert "attention_mask" in encoded_batch
    assert isinstance(actual_answers, list)
    assert actual_answers == ["4"]

    # Verify mask tokens are appended
    # num_mask_tokens = max_answer_length (3) + buffer (5) = 8
    expected_num_mask_tokens = len(answer_token_ids[0]) + mask_token_buffer
    expected_total_length = question_input_ids.shape[1] + expected_num_mask_tokens

    assert encoded_batch["input_ids"].shape[1] == expected_total_length
    assert encoded_batch["attention_mask"].shape[1] == expected_total_length

    # Verify the appended tokens are all mask tokens
    appended_tokens = encoded_batch["input_ids"][0, question_input_ids.shape[1] :]
    assert torch.all(appended_tokens == mask_token_id)

    # Verify attention mask is all 1s for the appended region
    appended_attention = encoded_batch["attention_mask"][0, question_input_ids.shape[1] :]
    assert torch.all(appended_attention == 1)

    # Verify tokenizer was called with correct arguments
    mock_tokenizer.apply_chat_template.assert_called_once()
    call_kwargs = mock_tokenizer.apply_chat_template.call_args.kwargs
    assert call_kwargs["add_generation_prompt"] is True

    # Verify tokenizer was called twice (once for question, once for answers)
    assert mock_tokenizer.call_count == 2

    # Verify first call (question) used left padding
    first_call_kwargs = mock_tokenizer.call_args_list[0].kwargs
    assert first_call_kwargs["padding_side"] == "left"

    # Verify second call (answers) used no special tokens
    second_call_kwargs = mock_tokenizer.call_args_list[1].kwargs
    assert second_call_kwargs["add_special_tokens"] is False
