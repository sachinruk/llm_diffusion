import dataclasses

import datasets
import peft
import torch
import transformers
import trl

from src import config, data


@dataclasses.dataclass
class ModelTrainerConfig:
    model: transformers.PreTrainedModel
    data_collator: data.CollateFn
    sft_config: trl.SFTConfig
    train_dataset: datasets.Dataset
    eval_dataset: datasets.Dataset


def get_model_and_tokenizer(
    hyper_parameters: config.HyperParameters, device: torch.device
) -> tuple[transformers.modeling_utils.PreTrainedModel, transformers.PreTrainedTokenizer]:
    llm_model: transformers.modeling_utils.PreTrainedModel = (
        transformers.AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=hyper_parameters.model,
            torch_dtype=torch.bfloat16 if device.type in {"cuda", "mps"} else torch.float32,
            attn_implementation="flash_attention_2" if device.type == "cuda" else "sdpa",
            device_map="auto",
        )
    )
    tokenizer: transformers.PreTrainedTokenizer = transformers.AutoTokenizer.from_pretrained(
        hyper_parameters.model
    )
    tokenizer.add_special_tokens({"additional_special_tokens": [config.MASK_TOKEN]})
    llm_model.resize_token_embeddings(len(tokenizer))
    return llm_model, tokenizer


def get_lora_model(
    model: transformers.modeling_utils.PreTrainedModel, hyper_parameters: config.HyperParameters
) -> transformers.modeling_utils.PreTrainedModel:
    # Configure LoRA
    lora_config = peft.LoraConfig(**hyper_parameters.lora_config.model_dump())

    # Apply LoRA to the model
    lora_model = peft.get_peft_model(model, lora_config)
    lora_model.print_trainable_parameters()
    return lora_model


def patch_causal_attention():
    if not hasattr(torch.nn.functional.scaled_dot_product_attention, "_is_patched"):
        original_sdpa = torch.nn.functional.scaled_dot_product_attention

    def universal_sdpa(
        query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None
    ):
        if attn_mask is not None:
            last_row = attn_mask[..., -1, :]
            universal_mask = last_row.unsqueeze(-2).expand_as(attn_mask)
            attn_mask = universal_mask

        return original_sdpa(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=False,
            scale=scale,
        )

    universal_sdpa._is_patched = True
    torch.nn.functional.scaled_dot_product_attention = universal_sdpa


def get_sft_config(hyper_parameters: config.HyperParameters) -> trl.SFTConfig:
    return trl.SFTConfig(
        output_dir=str(hyper_parameters.output_dir),
        num_train_epochs=hyper_parameters.epochs,
        per_device_train_batch_size=hyper_parameters.batch_size,
        gradient_accumulation_steps=hyper_parameters.gradient_accumulation_steps,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        optim="adamw_torch_fused",
        save_strategy="steps",
        save_steps=1 / hyper_parameters.log_frequency_per_epoch,
        eval_strategy="steps",  # must match save_strategy for load_best_model_at_end
        eval_steps=1 / hyper_parameters.log_frequency_per_epoch,
        load_best_model_at_end=True,  # important so that we can download the best model at the end
        learning_rate=hyper_parameters.learning_rate,
        bf16=torch.cuda.is_bf16_supported(),
        push_to_hub=False,
        report_to="wandb",
        dataset_kwargs={"skip_prepare_dataset": True},
        remove_unused_columns=False,
        group_by_length=False,
        dataloader_pin_memory=True,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=hyper_parameters.dataloader_num_workers,
        # use_liger_kernel=hyper_parameters.use_liger_kernel,
        # use_liger=hyper_parameters.use_liger_kernel,
    )


def get_trainer(
    model_trainer_config: ModelTrainerConfig,
    callbacks: list[transformers.TrainerCallback],
    hyper_parameters: config.HyperParameters,
) -> trl.SFTTrainer:
    lora_config = peft.LoraConfig(**hyper_parameters.lora_config.model_dump())

    return trl.SFTTrainer(
        model=model_trainer_config.model,
        args=model_trainer_config.sft_config,
        train_dataset=model_trainer_config.train_dataset,
        eval_dataset=model_trainer_config.eval_dataset,
        data_collator=model_trainer_config.data_collator,
        peft_config=lora_config,
        callbacks=callbacks,
    )
