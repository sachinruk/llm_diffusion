import datetime
import os
import random

import click
import numpy as np
import torch
import wandb
from loguru import logger

from src import config, data, evaluator, inference, model


def _wandb_init(hyper_parameters: config.HyperParameters):
    name = f"{hyper_parameters.wandb_config.project}-{datetime.datetime.now()}"
    project = hyper_parameters.wandb_config.project
    if hyper_parameters.debug:
        name = "debug-" + name
        project = "debug-" + project

    wandb.init(
        entity=hyper_parameters.wandb_config.entity,
        project=project,
        name=name,
        config=hyper_parameters.model_dump(),
        dir=hyper_parameters.wandb_config.wandb_log_path,
    )


def _setup_environment(hyper_parameters: config.HyperParameters, rank: int):
    # Seed all random number generators
    random.seed(hyper_parameters.seed + rank)
    np.random.seed(hyper_parameters.seed + rank)
    torch.manual_seed(hyper_parameters.seed + rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(hyper_parameters.seed + rank)

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    # Empty torch cache before starting
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.info("Cleared CUDA cache")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        torch.mps.empty_cache()
        logger.info("Cleared MPS cache")

    # Create output directories
    hyper_parameters.output_dir.mkdir(parents=True, exist_ok=True)
    hyper_parameters.wandb_config.wandb_log_path.mkdir(parents=True, exist_ok=True)


@click.command()
@click.option(
    "--hyper-parameters-json",
    default="{}",
    help="JSON string containing hyperparameters to override defaults",
)
def main(hyper_parameters_json: str):
    """
    Main training function for CLIP-JEPA model.

    Pass hyperparameters as a JSON string to override defaults.
    Example:
        python -m src.main --hyper-parameters-json '{"epochs": 10, "batch_size": 16}'
    """
    # Parse hyperparameters
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    if rank != 0:
        import logging

        logging.getLogger().setLevel(logging.ERROR)
        logger.disable("")  # Disable all loguru logging

    logger.info("Parsing hyperparameters...")
    hyper_parameters = config.HyperParameters.model_validate_json(hyper_parameters_json)
    if rank == 0:
        _wandb_init(hyper_parameters)
    logger.info(f"Hyperparameters: {hyper_parameters.model_dump_json(indent=2)}")
    _setup_environment(hyper_parameters, rank)
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)  # <-- critical
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    train_dataset, eval_dataset = data.load_dataset(hyper_parameters)
    llm_model, tokenizer = model.get_model_and_tokenizer(hyper_parameters, device, local_rank)
    model.patch_causal_attention()

    logger.info(
        f"[rank{local_rank}] visible={os.environ.get('CUDA_VISIBLE_DEVICES')} "
        f"acc_device_should_be=cuda:0  "
        f"hf_device_map={getattr(llm_model, 'hf_device_map', None)}  "
        f"is_4bit={getattr(llm_model, 'is_loaded_in_4bit', False)}"
    )

    model_trainer_config = model.ModelTrainerConfig(
        model=llm_model,
        sft_config=model.get_sft_config(hyper_parameters),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data.PretrainingCollateFn(tokenizer, max_length=hyper_parameters.max_length),
    )
    log_interval = (
        len(train_dataset)
        // hyper_parameters.batch_size
        // world_size
        // hyper_parameters.log_frequency_per_epoch
    )
    logger.info(f"Log interval: {log_interval}")
    callbacks = [
        evaluator.AccelerateEvalCallback(
            eval_dataset=model_trainer_config.eval_dataset,
            collate_fn=model_trainer_config.data_collator,
            mask_token_id=tokenizer.convert_tokens_to_ids(config.MASK_TOKEN),
            per_device_eval_batch_size=hyper_parameters.batch_size,
            log_interval=log_interval,
        ),
    ]
    trainer = model.get_trainer(model_trainer_config, callbacks, hyper_parameters)
    trainer.train()
    logger.info("Pretraining complete")

    # Change data collator for SFT phase and continue training the same LoRA
    trainer.data_collator = data.SFTCollateFn(tokenizer, max_length=hyper_parameters.max_length)
    trainer.train()
    logger.info("SFT complete")

    output = inference.diffusion_inference_stepwise(
        model=llm_model,
        batch=trainer.data_collator(train_dataset),
        mask_token_id=tokenizer.convert_tokens_to_ids(config.MASK_TOKEN),
        end_token_id=tokenizer.convert_tokens_to_ids(config.IM_START_TOKEN),
        steps=hyper_parameters.steps,
    )
    trainer.save_model(str(hyper_parameters.output_dir / "sft_model"))

    logger.info("Running accelerated inference on evaluation set")
    inference.run_eval_inference(
        model=llm_model,
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        hyper_parameters=hyper_parameters,
    )
    logger.info("Eval inference complete")


if __name__ == "__main__":
    main()
