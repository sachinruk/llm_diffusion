import datetime
import os

import click
import torch
from loguru import logger
import lightning as L
import wandb

from src import config, data, losses, model, trainer, vision_model


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
    hyper_parameters.lora_config.lora_weight_path.mkdir(parents=True, exist_ok=True)
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

    logger.info("done")


if __name__ == "__main__":
    main()
