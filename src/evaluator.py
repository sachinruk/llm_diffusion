# evaluator.py
from __future__ import annotations

import accelerate
import datasets
import torch
import transformers
import wandb
from loguru import logger
from torch.utils.data import DataLoader

from src import data


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
        dataloader = accelerator.prepare_data_loader(dataloader)
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

    totals: torch.Tensor = torch.tensor(
        [total_correct_mask, total_tokens_mask, total_correct_nonmask, total_tokens_nonmask],
        device=device,
        dtype=torch.long,
    )
    totals = accelerator.reduce(totals, reduction="sum") if accelerator is not None else totals
    total_correct_mask, total_tokens_mask, total_correct_nonmask, total_tokens_nonmask = map(
        int, totals.tolist()
    )

    mask_acc = total_correct_mask / total_tokens_mask
    nonmask_acc = total_correct_nonmask / total_tokens_nonmask
    model.train()
    return {"eval/non_mask_accuracy": float(nonmask_acc), "eval/mask_accuracy": float(mask_acc)}


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
        model = kwargs["model"]
        if (
            state.global_step % self.log_interval == 0
            and state.global_step != self.last_logged_step
        ):
            self.last_logged_step = state.global_step
            acc = accelerate.Accelerator()
            logger.info(f"Evaluating model on {len(self.eval_dataset)} rows")
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
                logger.info(f"Evaluation metrics: {logs}")
                wandb.log(logs)
