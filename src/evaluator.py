# evaluator.py
from __future__ import annotations

import accelerate
import datasets
import torch
import transformers
from torch.utils.data import DataLoader

from src import data

__all__ = ["evaluate_accelerate", "FourTimesSchedule", "AccelerateEvalCallback"]


@torch.inference_mode()
def evaluate_accelerate(
    model: transformers.PreTrainedModel,
    eval_dataset: datasets.Dataset,
    collate_fn: data.CollateFn,
    mask_token_id: int,
    per_device_eval_batch_size: int = 8,
    accelerator: accelerate.Accelerator | None = None,
) -> dict[str, float]:
    """
    Evaluate `model` on the full `eval_dataset` across processes (multi-GPU) via Accelerate.

    Returns:
        dict(non_mask_accuracy=float, mask_accuracy=float)
    """
    dataloader = DataLoader(
        eval_dataset,
        batch_size=per_device_eval_batch_size,
        shuffle=False,
        drop_last=False,
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False,
    )

    if accelerator is not None:
        model, dataloader = accelerator.prepare(model, dataloader)
        device: torch.device = accelerator.device  # type: ignore[assignment]
    else:
        device = model.device  # type: ignore[assignment]

    model.eval()

    total_correct_mask = 0
    total_tokens_mask = 0
    total_correct_nonmask = 0
    total_tokens_nonmask = 0

    for batch in dataloader:
        batch = batch.to(device)
        outputs = model(**batch)
        preds = outputs.logits.argmax(dim=-1)
        attention_mask = batch["attention_mask"].bool()
        mask_token_id_mask = batch["input_ids"] == mask_token_id
        non_mask_token_id_mask = (~mask_token_id_mask) & attention_mask
        correct = (preds == batch["labels"]) & attention_mask

        total_correct_mask += (correct & mask_token_id_mask).sum().item()
        total_tokens_mask += mask_token_id_mask.sum().item()
        total_correct_nonmask += (correct & non_mask_token_id_mask).sum().item()
        total_tokens_nonmask += non_mask_token_id_mask.sum().item()

        # correct = accelerator.pad_across_processes(correct, dim=1, pad_index=0)
        # mask_pos = accelerator.pad_across_processes(mask_pos, dim=1, pad_index=0)
        # nonmask_pos = accelerator.pad_across_processes(nonmask_pos, dim=1, pad_index=0)

        # correct_g = accelerator.gather_for_metrics(correct.int())
        # mask_pos_g = accelerator.gather_for_metrics(mask_pos.int())
        # nonmask_pos_g = accelerator.gather_for_metrics(nonmask_pos.int())

        # total_correct_mask += (correct_g.bool() & mask_pos_g.bool()).sum().item()
        # total_tokens_mask += mask_pos_g.sum().item()

        # total_correct_nonmask += (correct_g.bool() & nonmask_pos_g.bool()).sum().item()
        # total_tokens_nonmask += nonmask_pos_g.sum().item()

    # Merge across processes and broadcast the merged result
    if accelerator is not None and accelerator.num_processes > 1:
        # Gather confusion matrices from all processes
        all_total_correct_mask: list[None | int] = [None] * accelerator.num_processes
        all_total_tokens_mask: list[None | int] = [None] * accelerator.num_processes
        all_total_correct_nonmask: list[None | int] = [None] * accelerator.num_processes
        all_total_tokens_nonmask: list[None | int] = [None] * accelerator.num_processes
        torch.distributed.all_gather_object(all_total_correct_mask, total_correct_mask)
        torch.distributed.all_gather_object(all_total_tokens_mask, total_tokens_mask)
        torch.distributed.all_gather_object(all_total_correct_nonmask, total_correct_nonmask)
        torch.distributed.all_gather_object(all_total_tokens_nonmask, total_tokens_nonmask)

        total_correct_mask = sum(all_total_correct_mask)
        total_tokens_mask = sum(all_total_tokens_mask)
        total_correct_nonmask = sum(all_total_correct_nonmask)
        total_tokens_nonmask = sum(all_total_tokens_nonmask)

    mask_acc = total_correct_mask / total_tokens_mask
    nonmask_acc = total_correct_nonmask / total_tokens_nonmask
    return {"non_mask_accuracy": float(nonmask_acc), "mask_accuracy": float(mask_acc)}


class AccelerateEvalCallback(transformers.TrainerCallback):
    def __init__(
        self,
        eval_dataset,
        collate_fn,
        mask_token_id: int,
        per_device_eval_batch_size: int = 8,
        log_interval: int = 1000,
    ):
        super().__init__()
        self.eval_dataset = eval_dataset
        self.collate_fn = collate_fn
        self.mask_token_id = mask_token_id
        self.per_device_eval_batch_size = per_device_eval_batch_size
        self.log_interval = log_interval
        self.last_logged_step = -1

    def on_step_end(self, args, state, control, **kwargs):
        trainer = kwargs.get("trainer", None)
        model = kwargs.get("model", None)
        if trainer is None or model is None:
            return
        if (
            state.global_step % self.log_interval == 0
            and state.global_step != self.last_logged_step
        ):
            self.last_logged_step = state.global_step
            acc = getattr(trainer, "accelerator", None)
            metrics = evaluate_accelerate(
                model=model,
                eval_dataset=self.eval_dataset,
                collate_fn=self.collate_fn,
                mask_token_id=self.mask_token_id,
                per_device_eval_batch_size=self.per_device_eval_batch_size,
                accelerator=acc,
            )
            if acc is None or acc.is_main_process:
                logs = {**metrics, "step": int(state.global_step)}
                trainer.log(logs)
