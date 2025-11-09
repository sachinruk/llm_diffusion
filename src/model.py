import peft
import transformers
import torch

from src import config


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

def get_lora_model(model: transformers.modeling_utils.PreTrainedModel, hyper_parameters: config.HyperParameters) -> transformers.modeling_utils.PreTrainedModel:
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
