import torch
import transformers
import datasets
import wandb

from src import data


@torch.inference_mode()
def _calculate_accuracy(
    model: transformers.PreTrainedModel, batch: dict[str, torch.Tensor], mask_token_id: int
) -> tuple[float, float]:
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    attention_mask = batch["attention_mask"]

    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits
    predicted_ids = torch.argmax(logits, dim=-1)

    mask_token_id_mask = input_ids == mask_token_id
    non_mask_token_id_mask = (~mask_token_id_mask) & (attention_mask == 1)

    non_mask_accuracy = (
        (predicted_ids[non_mask_token_id_mask] == labels[non_mask_token_id_mask]).mean().item()
    )
    mask_accuracy = (predicted_ids[mask_token_id_mask] == labels[mask_token_id_mask]).mean().item()
    return non_mask_accuracy, mask_accuracy


class AccuracyCallback(transformers.TrainerCallback):
    def __init__(
        self,
        model: transformers.PreTrainedModel,
        dataset: datasets.Dataset,
        log_interval: int,
        collate_fn: data.CollateFn,
        batch_size: int,
    ):
        self.log_interval = log_interval
        self.model = model
        self.collate_fn = collate_fn
        self.dataset = dataset
        self.last_logged_step = 0
        self.batch_size = batch_size

    def on_step_end(self, args, state, control, **kwargs):
        # Log predictions every log_interval steps
        if (
            state.global_step > 0
            and state.global_step % self.log_interval == 0
            and state.global_step != self.last_logged_step
        ):
            sample_data = [
                self.dataset[i % len(self.dataset)]
                for i in range(state.global_step, state.global_step + self.batch_size)
            ]
            batch = self.collate_fn(sample_data)
            self.model.eval()
            non_mask_accuracy, mask_accuracy = _calculate_accuracy(
                self.model, batch, self.collate_fn.mask_token_id
            )
            self.model.train()
            wandb.log(
                {
                    "non_mask_accuracy": non_mask_accuracy,
                    "mask_accuracy": mask_accuracy,
                    "step": state.global_step,
                }
            )
