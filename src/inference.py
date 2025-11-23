import torch
import transformers


def _get_quotas(init_mask_counts: torch.Tensor, steps: int) -> torch.Tensor:
    """
    Split the initial number of masks per row across `steps`,
    distributing remainders to earlier steps.

    Args:
        init_mask_counts: [B] long tensor
        steps: number of diffusion steps

    Returns:
        quotas: [steps, B] long tensor
    """
    base = init_mask_counts // steps  # [B]
    rem = init_mask_counts % steps  # [B]
    return torch.stack(
        [(base + (rem > s).long()) for s in range(steps)],
        dim=0,  # [steps, B]
    )


def _compute_maskable_area(input_ids: torch.Tensor, end_token_id: int) -> torch.Tensor:
    """
    Compute a boolean mask of positions strictly before the first EOS
    in each row.

    Args:
        input_ids: [B, L]
        end_token_id: EOS token id

    Returns:
        maskable_area: [B, L] bool tensor, True before first EOS, False at/after.
    """
    eos_hits: torch.Tensor = input_ids == end_token_id
    # cumsum over EOS hits is 0 until first EOS, then >=1
    return ~(eos_hits.cumsum(dim=1).bool())


def _compute_allowed_masks(
    input_ids: torch.Tensor,
    mask_token_id: int,
    end_token_id: int,
) -> torch.Tensor:
    """
    Positions that can be filled this step: mask tokens before EOS.

    Args:
        input_ids: [B, L]
        mask_token_id: mask token id
        end_token_id: EOS token id

    Returns:
        allowed_masks: [B, L] bool
    """
    maskable_area: torch.Tensor = _compute_maskable_area(input_ids, end_token_id)  # [B, L]
    masked_tokens: torch.Tensor = input_ids == mask_token_id
    return maskable_area & masked_tokens


def compute_step_updates(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    step_quota: torch.Tensor,
    mask_token_id: int,
    end_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    For a single diffusion step, determine which positions to unmask,
    in a fully vectorized way.

    Args:
        logits: [B, L, V]
        input_ids: [B, L]
        step_quota: [B] long tensor indicating how many tokens to unmask per row.
        mask_token_id: mask token id
        end_token_id: EOS token id

    Returns:
        batch_indices: [N] indices of rows to update.
        pos_indices:   [N] positions within rows to update.
        new_tokens:    [N] token ids to write at those positions.
    """
    device = input_ids.device
    batch_size, seq_len, _ = logits.shape

    # [B, L], [B, L]
    confidence, pred_tok = logits.max(dim=-1)

    # Allowed positions (masked & before EOS)
    allowed_masks: torch.Tensor = _compute_allowed_masks(
        input_ids, mask_token_id, end_token_id
    )  # [B, L]

    # How many mask positions remain in each row
    remaining_per_row: torch.Tensor = allowed_masks.sum(dim=-1)  # [B]
    # Clamp quota by how many masks actually remain in that row
    step_quota_clamped: torch.Tensor = step_quota.clamp(max=remaining_per_row)  # [B]
    k_max: int = int(step_quota_clamped.max().item())

    if k_max == 0:
        # No quota for any row this step
        empty = input_ids.new_empty(0, dtype=torch.long)
        return empty, empty, empty

    # Prevent selecting disallowed positions
    masked_confidence: torch.Tensor = confidence.clone()
    masked_confidence[~allowed_masks] = -torch.inf

    # Take top-k_max per row
    _, topk_idx = masked_confidence.topk(k=k_max, dim=-1)  # [B, k_max]
    topk_tokens: torch.Tensor = pred_tok.gather(dim=1, index=topk_idx)  # [B, k_max]

    # Build a mask over [B, k_max] selecting only first k_b entries
    arange_k: torch.Tensor = torch.arange(k_max, device=device).unsqueeze(0)  # [1, k_max]
    step_quota_expanded: torch.Tensor = step_quota_clamped.unsqueeze(1)  # [B, 1]
    update_mask: torch.Tensor = arange_k < step_quota_expanded  # [B, k_max]

    # Flatten to 1D updates
    batch_indices_full: torch.Tensor = (
        torch.arange(batch_size, device=device).unsqueeze(1).expand(-1, k_max)
    )  # [B, k_max]

    batch_indices: torch.Tensor = batch_indices_full[update_mask]  # [N]
    pos_indices: torch.Tensor = topk_idx[update_mask]  # [N]
    new_tokens: torch.Tensor = topk_tokens[update_mask]  # [N]

    return batch_indices, pos_indices, new_tokens


@torch.inference_mode()
def diffusion_inference_stepwise(
    model: transformers.PreTrainedModel,
    batch: transformers.BatchEncoding,
    mask_token_id: int,
    end_token_id: int,
    steps: int,
) -> transformers.BatchEncoding:
    """
    Pure functional diffusion-style infilling over `steps` with no per-batch loops.

    Args:
        model: HF model
        batch: BatchEncoding with "input_ids" and "attention_mask"
        mask_token_id: mask token id
        end_token_id: EOS token id
        steps: number of diffusion steps

    Returns:
        BatchEncoding with updated "input_ids" and original "attention_mask".
    """
    was_training: bool = model.training
    model.eval()

    batch = batch.to(model.device)
    input_ids: torch.Tensor = batch["input_ids"].clone()
    attention_mask: torch.Tensor = batch["attention_mask"]
    device = input_ids.device

    # ---- initial quotas (no per-row Python loop) ----
    init_mask_counts: torch.Tensor = _compute_allowed_masks(
        input_ids, mask_token_id, end_token_id
    ).sum(dim=-1)  # [B]
    quotas: torch.Tensor = _get_quotas(init_mask_counts, steps).to(device=device)  # [steps, B]

    # ---- diffusion steps ----
    for step in range(steps):
        # Optional global early exit if no masks remain
        if not (input_ids == mask_token_id).any():
            break

        logits: torch.Tensor = model(input_ids=input_ids, attention_mask=attention_mask).logits

        step_quota: torch.Tensor = quotas[step]  # [B]
        (
            batch_indices,
            pos_indices,
            new_tokens,
        ) = compute_step_updates(
            logits=logits,
            input_ids=input_ids,
            step_quota=step_quota,
            mask_token_id=mask_token_id,
            end_token_id=end_token_id,
        )

        # If nothing to update this step and no masks remain, we can exit early
        if batch_indices.numel() == 0:
            if not (input_ids == mask_token_id).any():
                break
            continue

        input_ids[batch_indices, pos_indices] = new_tokens

    if was_training:
        model.train()

    return transformers.BatchEncoding(
        {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
    )


# Optional: keep a tiny OO wrapper that just calls the functional version.
class DiffusionInference:
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        mask_token_id: int,
        end_token_id: int,
        steps: int,
    ) -> None:
        self.model = model
        self.mask_token_id = mask_token_id
        self.end_token_id = end_token_id
        self.steps = steps

    @torch.inference_mode()
    def __call__(self, batch: transformers.BatchEncoding) -> transformers.BatchEncoding:
        return diffusion_inference_stepwise(
            model=self.model,
            batch=batch,
            mask_token_id=self.mask_token_id,
            end_token_id=self.end_token_id,
            steps=self.steps,
        )
