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


def test__compute_maskable_area():
    """Test computing maskable area before second EOS token."""
    # Test with EOS at different positions
    # Assume EOS token id = 2
    end_token_id = 2

    # Batch with EOS at different positions
    # The function returns True for positions before the SECOND EOS
    input_ids = torch.tensor(
        [
            [1, 5, 3, 2, 7, 8],  # One EOS at position 3 -> all True (no second EOS)
            [4, 2, 9, 1, 2, 3],  # EOS at positions 1 and 4 -> False at/after position 4
            [6, 7, 8, 9, 1, 3],  # No EOS -> all True
        ]
    )

    result = inference._compute_maskable_area(input_ids, end_token_id)

    expected = torch.tensor(
        [
            [True, True, True, True, True, True],  # Only 1 EOS, all before "second"
            [True, True, True, True, False, False],  # Second EOS at position 4
            [True, True, True, True, True, True],  # No EOS at all
        ]
    )

    assert torch.all(result == expected)
    assert result.shape == input_ids.shape
    assert result.dtype == torch.bool


def test__compute_allowed_masks():
    """Test computing positions that can be filled (masked & before second EOS)."""
    mask_token_id = 99
    end_token_id = 2

    # Positions are allowed if: (1) they are mask tokens AND (2) before second EOS
    input_ids = torch.tensor(
        [
            [
                99,
                99,
                3,
                2,
                99,
                2,
            ],  # Two EOS: positions 3 and 5 -> masks at 0,1,4 allowed (before pos 5)
            [
                4,
                2,
                99,
                99,
                2,
                99,
            ],  # Two EOS: positions 1 and 4 -> masks at 2,3 allowed (before pos 4)
            [99, 5, 99, 99, 1, 3],  # No EOS -> all masks allowed (positions 0, 2, 3)
        ]
    )

    result = inference._compute_allowed_masks(input_ids, mask_token_id, end_token_id)

    expected = torch.tensor(
        [
            [True, True, False, False, True, False],  # masks at 0,1,4 before second EOS at 5
            [False, False, True, True, False, False],  # masks at 2,3 before second EOS at 4
            [True, False, True, True, False, False],  # all masks allowed (no second EOS)
        ]
    )

    assert torch.all(result == expected)
    assert result.shape == input_ids.shape
    assert result.dtype == torch.bool


def test_compute_step_updates():
    """Test computing which positions to unmask in a diffusion step."""
    mask_token_id = 99
    end_token_id = 2

    # Create input with masks (using two EOS to define the maskable region)
    input_ids = torch.tensor(
        [
            [2, 99, 99, 99, 2, 6],  # First EOS at 0, masks at 1,2,3, second EOS at 4
            [2, 1, 99, 99, 2, 7],  # First EOS at 0, masks at 2,3, second EOS at 4
        ]
    )

    allowed_masks = inference._compute_allowed_masks(input_ids, mask_token_id, end_token_id)

    assert torch.all(
        allowed_masks
        == torch.tensor(
            [
                [False, True, True, True, False, False],  # 3 masks before second EOS
                [False, False, True, True, False, False],  # 2 masks before second EOS
            ]
        )
    )

    # Create logits with deterministic confidence values
    batch_size, seq_len = input_ids.shape
    vocab_size = 50
    logits = torch.randn(batch_size, seq_len, vocab_size)

    # Set high confidence for specific positions and tokens
    # Row 0, position 1: high confidence for token 10
    logits[0, 1, :] = -10
    logits[0, 1, 10] = 5.0
    # Row 0, position 2: medium confidence for token 11
    logits[0, 2, :] = -10
    logits[0, 2, 11] = 3.0
    # Row 0, position 3: low confidence for token 12
    logits[0, 3, :] = -10
    logits[0, 3, 12] = 1.0

    # Row 1, position 2: high confidence for token 20
    logits[1, 2, :] = -10
    logits[1, 2, 20] = 4.0
    # Row 1, position 3: medium confidence for token 21
    logits[1, 3, :] = -10
    logits[1, 3, 21] = 2.0

    # Request 2 updates for row 0, 1 update for row 1
    step_quota = torch.tensor([2, 1])

    batch_indices, pos_indices, new_tokens = inference.compute_step_updates(
        logits, input_ids, allowed_masks, step_quota
    )

    # Should unmask 2 positions in row 0 (highest confidence: pos 1 and 2)
    # Should unmask 1 position in row 1 (highest confidence: pos 2)
    assert len(batch_indices) == 3
    assert len(pos_indices) == 3
    assert len(new_tokens) == 3

    # Check row 0 updates (positions 1 and 2 with highest confidence)
    row0_mask = batch_indices == 0
    row0_positions = pos_indices[row0_mask]
    row0_tokens = new_tokens[row0_mask]

    assert row0_mask.sum() == 2
    assert 1 in row0_positions
    assert 2 in row0_positions
    assert 10 in row0_tokens
    assert 11 in row0_tokens

    # Check row 1 updates (position 2 with highest confidence)
    row1_mask = batch_indices == 1
    assert row1_mask.sum() == 1
    assert pos_indices[row1_mask].item() == 2
    assert new_tokens[row1_mask].item() == 20


def test_compute_step_updates_empty():
    """Test compute_step_updates with no masks or zero quota."""
    mask_token_id = 99
    end_token_id = 2
    vocab_size = 50

    # No masks remaining (two EOS tokens to define boundary)
    input_ids = torch.tensor([[2, 1, 5, 3, 2, 8]])
    logits = torch.randn(1, 6, vocab_size)
    step_quota = torch.tensor([2])
    allowed_masks = inference._compute_allowed_masks(input_ids, mask_token_id, end_token_id)

    batch_indices, pos_indices, new_tokens = inference.compute_step_updates(
        logits, input_ids, allowed_masks, step_quota
    )

    assert len(batch_indices) == 0
    assert len(pos_indices) == 0
    assert len(new_tokens) == 0

    # Zero quota (two EOS tokens to define boundary)
    input_ids = torch.tensor([[2, 99, 99, 3, 2, 8]])
    logits = torch.randn(1, 6, vocab_size)
    step_quota = torch.tensor([0])
    allowed_masks = inference._compute_allowed_masks(input_ids, mask_token_id, end_token_id)

    batch_indices, pos_indices, new_tokens = inference.compute_step_updates(
        logits, input_ids, allowed_masks, step_quota
    )

    assert len(batch_indices) == 0
    assert len(pos_indices) == 0
    assert len(new_tokens) == 0
