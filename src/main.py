import datetime
import os

import click
import lightning as L
import torch
import wandb
from loguru import logger

from src import config, data, model


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


def _setup_environment(hyper_parameters: config.HyperParameters):
    L.seed_everything(hyper_parameters.seed)
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
    logger.info("Parsing hyperparameters...")
    hyper_parameters = config.HyperParameters.model_validate_json(hyper_parameters_json)
    _wandb_init(hyper_parameters)
    logger.info(f"Hyperparameters: {hyper_parameters.model_dump_json(indent=2)}")
    _setup_environment(hyper_parameters)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    logger.info(f"Using device: {device}")

    train_dataset, eval_dataset = data.load_dataset(hyper_parameters)
    llm_model, tokenizer = model.get_model_and_tokenizer(hyper_parameters, device)
    model.patch_causal_attention()

    model_trainer_config = model.ModelTrainerConfig(
        model=llm_model,
        sft_config=model.get_sft_config(hyper_parameters),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data.PretrainingCollateFn(tokenizer),
    )
    trainer = model.get_trainer(model_trainer_config, hyper_parameters)
    trainer.train()
    logger.info("Pretraining complete")

    model_trainer_config.data_collator = data.SFTCollateFn(tokenizer)
    trainer = model.get_trainer(model_trainer_config, hyper_parameters)
    trainer.train()
    logger.info("SFT complete")


if __name__ == "__main__":
    main()
