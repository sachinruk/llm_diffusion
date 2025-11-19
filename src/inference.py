import torch
import transformers


def _first_idx(row_input_ids: torch.Tensor, end_token_id: int) -> int:
    max_length = row_input_ids.shape[0]
    pos = (row_input_ids == end_token_id).nonzero(as_tuple=False)
    return pos[0].item() if pos.numel() > 0 else max_length


def _get_quotas(init_mask_counts: torch.Tensor, steps: int) -> torch.Tensor:
    base = init_mask_counts // steps
    rem = init_mask_counts % steps
    return torch.stack(
        [(base + (rem > s).long()) for s in range(steps)],
        dim=0,
    )


class DiffusionInference:
    """
    Iteratively infills mask tokens over `steps`. For each batch row:
      - Compute the initial number of masks *before* the first EOS (if any).
      - Divide that count across steps (distributing remainders to earlier steps).
      - At each step, choose the top-k masked positions by confidence (max logit)
        and fill only those with their argmax token.
      - On the final step, unmask all remaining (before EOS).
    Positions after the first EOS (pre-existing or newly predicted) are ignored.
    """

    def __init__(
        self,
        model: transformers.PreTrainedModel,
        mask_token_id: int,
        end_token_id: int,
        steps: int,
    ):
        assert steps >= 1, "steps must be >= 1"
        self.model = model
        self.mask_token_id = mask_token_id
        self.end_token_id = end_token_id
        self.steps = steps

    @torch.inference_mode()
    def __call__(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        was_training = self.model.training
        self.model.eval()

        # Work on copies to avoid mutating the original batch in-place.
        input_ids = batch["input_ids"].clone()
        attention_mask = batch["attention_mask"]

        device = self.model.device
        batch_size, L = input_ids.shape

        # Initial EOS boundaries and initial mask counts (only up to EOS!)
        eos_idx = torch.tensor(
            [_first_idx(input_ids[b], self.end_token_id) for b in range(batch_size)]
        ).to(device)
        init_mask_counts = torch.tensor(
            [(input_ids[b, : eos_idx[b]] == self.mask_token_id).sum() for b in range(batch_size)]
        ).to(device)

        quotas = _get_quotas(init_mask_counts, self.steps)

        for s in range(self.steps):
            # Forward pass
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
            logits = outputs.logits  # [B, L, V]

            # For each row, pick top-k masked positions by confidence (max logit).
            for b in range(batch_size):
                # Recompute EOS boundary each step (it may have been newly predicted).
                e = _first_idx(input_ids[b], self.end_token_id, L)

                # Masked positions strictly before EOS
                mask_pos = (
                    (input_ids[b, :e] == self.mask_token_id).nonzero(as_tuple=False).squeeze(-1)
                )
                if mask_pos.numel() == 0:
                    continue

                k = int(min(quotas[s, b].item(), mask_pos.numel()))

                # Compute per-position confidence and predicted token.
                # conf: [num_masks], pred_tok: [num_masks]
                per_pos_logits = logits[b, mask_pos, :]  # [num_masks, V]
                confidence, pred_tok = per_pos_logits.max(
                    dim=-1
                )  # argmax token + its logit, size: [num_masks]

                # Select top-k positions by confidence.
                topk_idx = confidence.topk(k=k, dim=0).indices  # [k]
                pos_to_update = mask_pos[topk_idx]  # [k]
                # Overwrite selected masked positions with their predicted tokens.
                input_ids[b, pos_to_update] = pred_tok[topk_idx]  # [k]

            # Optional early exit if nothing masked remains anywhere.
            if not (input_ids == self.mask_token_id).any():
                break

        if was_training:
            self.model.train()

        return {"input_ids": input_ids, "attention_mask": attention_mask}
