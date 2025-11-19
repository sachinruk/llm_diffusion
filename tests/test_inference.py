import torch

from src import inference


def test__get_quotas_distributions():
    """Test quota distribution across various scenarios in a batch."""
    # Test multiple scenarios simultaneously:
    # Item 0: 10 masks, 5 steps -> even distribution [2, 2, 2, 2, 2]
    # Item 1: 10 masks, 3 steps -> remainder=1 [4, 3, 3]
    # Item 2: 11 masks, 4 steps -> remainder=3 [3, 3, 3, 2]
    # Item 3: 7 masks, 3 steps -> remainder=1 [3, 2, 2]

    # Even distribution
    init_mask_counts = torch.tensor([10])
    steps = 5
    result = inference._get_quotas(init_mask_counts, steps)
    expected = torch.tensor([[2], [2], [2], [2], [2]])
    assert torch.all(result == expected)
    assert result.sum() == init_mask_counts.sum()

    # Single and multiple remainders in batch
    init_mask_counts = torch.tensor([10, 11, 7])
    steps = 3
    result = inference._get_quotas(init_mask_counts, steps)
    # Item 0: 10 -> [4, 3, 3]
    # Item 1: 11 -> [4, 4, 3]
    # Item 2: 7 -> [3, 2, 2]
    expected = torch.tensor([[4, 4, 3], [3, 4, 2], [3, 3, 2]])
    assert torch.all(result == expected)
    assert torch.all(result.sum(dim=0) == init_mask_counts)


def test__get_quotas_edge_cases():
    """Test edge cases: zero masks, single step, more steps than masks."""
    # Zero masks mixed with non-zero
    init_mask_counts = torch.tensor([0, 5, 0])
    steps = 3
    result = inference._get_quotas(init_mask_counts, steps)
    expected = torch.tensor([[0, 2, 0], [0, 2, 0], [0, 1, 0]])
    assert torch.all(result == expected)
    assert torch.all(result.sum(dim=0) == init_mask_counts)

    # Single step (all at once)
    init_mask_counts = torch.tensor([15, 8])
    steps = 1
    result = inference._get_quotas(init_mask_counts, steps)
    expected = torch.tensor([[15, 8]])
    assert torch.all(result == expected)

    # More steps than masks
    init_mask_counts = torch.tensor([2])
    steps = 5
    result = inference._get_quotas(init_mask_counts, steps)
    expected = torch.tensor([[1], [1], [0], [0], [0]])
    assert torch.all(result == expected)
    assert result.sum() == init_mask_counts.sum()

    init_mask_counts = torch.tensor([17, 23, 8, 42, 5])
    steps = 7
    result = inference._get_quotas(init_mask_counts, steps)

    # Correct output shape
    assert result.shape == (steps, len(init_mask_counts))
    # Conservation: sum across steps equals initial counts
    assert torch.all(result.sum(dim=0) == init_mask_counts)


def test__first_idx_found():
    """Test first_idx when end_token_id is found at various positions."""
    end_token_id = 99

    # End token at beginning
    row_ids = torch.tensor([99, 1, 2, 3, 4])
    assert inference._first_idx(row_ids, end_token_id) == 0

    # End token in middle
    row_ids = torch.tensor([1, 2, 99, 4, 5])
    assert inference._first_idx(row_ids, end_token_id) == 2

    # Multiple end tokens (should return first occurrence)
    row_ids = torch.tensor([1, 99, 3, 99, 5])
    assert inference._first_idx(row_ids, end_token_id) == 1


def test__first_idx_not_found():
    """Test first_idx when end_token_id is not found (returns max_length)."""
    end_token_id = 99

    # No end token
    row_ids = torch.tensor([1, 2, 3, 4, 5])
    assert inference._first_idx(row_ids, end_token_id) == len(row_ids)

    # Empty tensor
    row_ids = torch.tensor([])
    assert inference._first_idx(row_ids, end_token_id) == 0
