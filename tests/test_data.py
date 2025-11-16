import random

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
