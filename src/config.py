import pydantic
import torch
import pathlib

MASK_TOKEN = "<|mask|>"
IM_START_TOKEN = "<|im_start|>"

device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class WandbConfig(pydantic.BaseModel):
    project: str = "llm-diffusion"
    entity: str = "sachinruk"
    wandb_log_path: pathlib.Path = pathlib.Path("/tmp/wandb")


class LoraConfig(pydantic.BaseModel):
    r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: list[str] = [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ]
    bias: str = "none"


class HyperParameters(pydantic.BaseModel):
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset: str = "allenai/tulu-3-sft-mixture-0225"
    debug: bool = False
    seed: int = 42
    lr: float = 2e-4
    epochs: int = 1
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    log_interval: int = 1000
    lora_config: LoraConfig = LoraConfig()
    wandb_config: WandbConfig = WandbConfig()
