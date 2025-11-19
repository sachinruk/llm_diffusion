import pathlib

import pydantic
import torch

MASK_TOKEN = "<|mask|>"
IM_START_TOKEN = "<|im_start|>"

device: torch.device = torch.device(
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)


class WandbConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    project: str = "llm-diffusion"
    entity: str = "sachinruk"
    wandb_log_path: pathlib.Path = pathlib.Path("/tmp/wandb")


class QLoRAConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
    load_in_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"  # or "fp4"
    bnb_4bit_use_double_quant: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"  # use "float16" if your GPU lacks bf16


class LoraConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(extra="forbid")
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
    model_config = pydantic.ConfigDict(extra="forbid")
    model: str = "Qwen/Qwen3-4B-Instruct-2507"
    dataset: str = "allenai/tulu-3-sft-mixture-0225"
    debug: bool = False
    seed: int = 42
    learning_rate: float = 2e-4
    test_size: float = 0.1
    dataloader_num_workers: int = 4
    pin_memory: bool = True
    output_dir: pathlib.Path = pathlib.Path("/tmp/output")
    epochs: int = 1
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    log_interval: int = 1000
    lora_config: LoraConfig = LoraConfig()
    wandb_config: WandbConfig = WandbConfig()
    log_frequency_per_epoch: int = 4

    use_qlora: bool = False
    qlora_config: QLoRAConfig = QLoRAConfig()

    max_length: int = 768
