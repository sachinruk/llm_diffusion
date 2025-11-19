# LLM Diffusion

A PyTorch implementation of diffusion-based language model training using mask-and-denoise pretraining followed by supervised fine-tuning. The model uses QLoRA for efficient fine-tuning of large language models.

## Overview

This project implements a diffusion-inspired approach to language model training:
1. **Pretraining Phase**: Random tokens are masked, and the model learns to predict the original tokens
2. **SFT Phase**: Standard supervised fine-tuning on instruction-following data
3. **Iterative Inference**: At inference time, the model starts with a fully masked sequence and iteratively unmasks the most confident tokens

The architecture uses:
- **Base Model**: Qwen3-4B-Instruct-2507 (or any causal LLM)
- **Efficient Training**: Optional QLoRA (4-bit quantization + LoRA) for memory-efficient fine-tuning
- **Hardware Flexibility**: Automatically detects and uses CUDA, Apple Silicon (MPS), or CPU
- **Training Framework**: TRL (Transformers Reinforcement Learning) with Hugging Face integration

## Features

- **Mask-and-Denoise Pretraining**: Random masking with variable mask probability (10-95%)
- **Two-Phase Training**: Pretraining followed by supervised fine-tuning with the same LoRA adapters
- **Iterative Diffusion Inference**: Progressive unmasking based on model confidence
- **Efficient Fine-tuning**: QLoRA support for training large models with limited resources
- **Gradient Checkpointing**: Memory-efficient training for large models
- **Weights & Biases Integration**: Built-in experiment tracking
- **Prefix Masking for SFT**: Only compute loss on assistant responses, not user prompts

## Architecture

### Training Process

**Pretraining Phase:**
1. Random tokens in the input sequence are masked with probability p ∈ [0.1, 0.95]
2. The model predicts the original tokens at masked positions
3. Loss is computed on all non-padding tokens
4. LoRA adapters are trained to adapt the frozen base model

**SFT Phase:**
1. Input sequences contain instruction-response pairs
2. Only the assistant response is used for loss computation (prefix masking)
3. Random masking is still applied to the response portion
4. The same LoRA adapters continue training

### Inference Process

The `DiffusionInference` class implements iterative unmasking:

1. Start with a sequence where target tokens are masked
2. For each diffusion step:
   - Forward pass through the model
   - Compute confidence (max logit) for each masked position
   - Unmask the top-k most confident positions
3. Continue until all tokens are unmasked or EOS is generated

This allows the model to generate text by progressively refining a masked sequence, similar to diffusion models in computer vision.

## Installation

### Using the setup script (Recommended)

```bash
# Create virtual environment and install dependencies
./setup.sh

# Update existing environment
./setup.sh --update
```

### Manual installation

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Requirements

- Python 3.10+
- PyTorch 2.8.0+
- transformers (Hugging Face)
- peft (LoRA/QLoRA)
- trl (Transformers Reinforcement Learning)
- datasets (Hugging Face)
- accelerate
- bitsandbytes (for QLoRA quantization)
- wandb
- loguru
- click

See `requirements.txt` for the complete list.

## Usage

### Basic Training

```bash
python -m src.main
```

This will:
1. Load the default configuration
2. Run pretraining phase with mask-and-denoise objective
3. Automatically switch to SFT phase
4. Save the final model with LoRA adapters

### Custom Hyperparameters

Pass hyperparameters as a JSON string:

```bash
python -m src.main --hyper-parameters-json '{
  "epochs": 2,
  "batch_size": 8,
  "learning_rate": 2e-4,
  "use_qlora": true,
  "debug": false
}'
```

### Configuration Options

All hyperparameters can be configured via JSON. Key options include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `model` | `"Qwen/Qwen3-4B-Instruct-2507"` | Base language model |
| `dataset` | `"allenai/tulu-3-sft-mixture-0225"` | Training dataset |
| `epochs` | 1 | Number of training epochs (per phase) |
| `batch_size` | 16 | Training batch size per device |
| `learning_rate` | 2e-4 | Learning rate for LoRA adapters |
| `gradient_accumulation_steps` | 4 | Gradient accumulation steps |
| `use_qlora` | false | Enable 4-bit quantization |
| `max_length` | 768 | Maximum sequence length |
| `debug` | false | Debug mode (limits data to 100 examples) |

See `src/config.py` for all available configuration options.

### Using config.yaml

You can also modify `config.yaml` for easier configuration:

```yaml
debug: false
batch_size: 4
epochs: 2
learning_rate: 1.0e-3
use_qlora: false
dataloader_num_workers: 8

lora_config:
  r: 8
  lora_alpha: 16
```

## Configuration Details

### LoRA Settings

```python
lora_config:
  r: 16                    # LoRA rank
  lora_alpha: 32           # LoRA alpha (scaling factor)
  lora_dropout: 0.05       # Dropout probability
  target_modules:          # Modules to apply LoRA to
    - "q_proj"
    - "k_proj"
    - "v_proj"
    - "o_proj"
    - "gate_proj"
    - "up_proj"
    - "down_proj"
  bias: "none"
```

### QLoRA Settings

```python
qlora_config:
  load_in_4bit: true
  bnb_4bit_quant_type: "nf4"
  bnb_4bit_use_double_quant: true
  bnb_4bit_compute_dtype: "bfloat16"
```

## Dataset Format

The training expects datasets in Hugging Face format with a `messages` field containing conversation-style data:

```json
{
  "messages": [
    {"role": "user", "content": "What is the capital of France?"},
    {"role": "assistant", "content": "The capital of France is Paris."}
  ]
}
```

Default dataset: `allenai/tulu-3-sft-mixture-0225`

Compatible datasets should have:
- A `messages` field with conversation turns
- Proper role tags (`user`, `assistant`, `system`)

## Training Outputs

Training artifacts are saved to:
- **Model checkpoints**: `/tmp/output/` (configurable via `output_dir`)
- **LoRA adapters**: Automatically saved with checkpoints
- **Final model**: `/tmp/output/sft_model/` (after both training phases)
- **Wandb logs**: `/tmp/wandb/` (configurable via `wandb_log_path`)

## Monitoring

Training metrics are logged to Weights & Biases:

- **Loss**: Training and validation loss per step
- **Perplexity**: Model perplexity on masked token prediction
- **Custom Metrics**: Via callbacks (see `src/evaluator.py`)

Configure W&B settings:

```python
wandb_config:
  project: "llm-diffusion"
  entity: "your-wandb-username"
  wandb_log_path: "/tmp/wandb"
```

Set `debug: true` to prefix runs with "debug-" for easier filtering.

## Project Structure

```
llm_diffusion/
├── src/
│   ├── callbacks.py      # Training callbacks
│   ├── config.py         # Configuration classes and hyperparameters
│   ├── data.py           # Dataset loading and collate functions
│   ├── evaluator.py      # Evaluation callbacks and metrics
│   ├── inference.py      # Diffusion inference (iterative unmasking)
│   ├── main.py           # Training entry point
│   └── model.py          # Model loading and LoRA configuration
├── tests/
│   ├── test_data.py      # Data loading tests
│   └── test_inference.py # Inference tests
├── notebooks/
│   └── 2025-11-02-diffusion-llm.ipynb
├── config.yaml           # Example YAML configuration
├── requirements.txt      # Python dependencies
├── setup.sh              # Environment setup script
├── run.sh                # Training script with various configurations
└── README.md
```

## Hardware Requirements

- **Minimum**: CPU (very slow training, not recommended)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM
- **With QLoRA**: Can train on GPUs with 12GB+ VRAM using 4-bit quantization
- **Apple Silicon**: MPS backend supported for M1/M2/M3 Macs (without quantization)

The model automatically selects the best available device. With QLoRA (4-bit quantization), memory requirements are significantly reduced.

## Advanced Usage

### Using Different Models

```bash
python -m src.main --hyper-parameters-json '{
  "model": "meta-llama/Llama-3.2-3B-Instruct"
}'
```

Any causal language model from Hugging Face should work.

### Adjusting Mask Probability

The mask probability is randomly sampled per batch between 10% and 95%. This is controlled in the collate functions:

```python
min_mask_probability: float = 0.1
max_mask_probability: float = 0.95
```

### Inference with Diffusion

```python
from src.inference import DiffusionInference
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("your-model")
tokenizer = AutoTokenizer.from_pretrained("your-model")
mask_token_id = tokenizer.convert_tokens_to_ids("<|mask|>")

inference = DiffusionInference(
    model=model,
    mask_token_id=mask_token_id,
    end_token_id=tokenizer.eos_token_id,
    steps=10  # Number of iterative unmasking steps
)

# Create a batch with masked tokens
batch = tokenizer(["<|mask|> <|mask|> <|mask|>"], return_tensors="pt")
output = inference(batch)
```

## Technical Details

### Attention Mask Patching

The code patches PyTorch's scaled dot product attention to enable universal attention (all tokens can attend to all tokens, including future tokens). This is crucial for the mask-and-denoise objective:

```python
model.patch_causal_attention()
```

This modifies the attention mechanism to use the last row of the attention mask for all positions, effectively removing causal masking.

### Prefix Masking for SFT

During supervised fine-tuning, the loss is only computed on the assistant's response, not the user's prompt. This is achieved by:

1. Finding the last `<|im_start|>` token (marks the assistant's turn)
2. Masking all labels before that position with `-100` (ignored by CrossEntropyLoss)
3. Only masked tokens in the response are used for training

### Gradient Checkpointing

The trainer uses gradient checkpointing to reduce memory usage:

```python
gradient_checkpointing=True
gradient_checkpointing_kwargs={"use_reentrant": False}
```

This trades computation for memory, allowing larger batch sizes.

## Development

### Running Tests

```bash
# Test data loading
pytest tests/test_data.py

# Test inference
pytest tests/test_inference.py

# Test in debug mode
python -m src.main --hyper-parameters-json '{"debug": true}'
```

### Notebooks

Explore the notebooks for experimentation:
- `notebooks/2025-11-02-diffusion-llm.ipynb` - Model testing and visualization

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size` (e.g., 4 or 8)
- Increase `gradient_accumulation_steps` (e.g., 8 or 16)
- Enable `use_qlora: true` for 4-bit quantization
- Reduce `max_length` (e.g., 512 instead of 768)

### MPS (Apple Silicon) Issues

- QLoRA is not supported on MPS (CUDA only)
- Set `use_qlora: false` for Apple Silicon
- Some operations may fall back to CPU automatically
- Use smaller models (e.g., 1B-3B parameters)

### W&B Login Required

```bash
wandb login
```

Or set environment variable:
```bash
export WANDB_API_KEY=your_api_key
```

### Import Errors

Make sure you've activated the virtual environment:
```bash
source .venv/bin/activate  # On Unix/macOS
# or
.venv\Scripts\activate     # On Windows
```

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation if applicable]
```

## Acknowledgments

- Built on [Qwen3](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) from Alibaba Cloud
- Training framework: [TRL](https://github.com/huggingface/trl) (Transformers Reinforcement Learning)
- LoRA/QLoRA implementation: [PEFT](https://github.com/huggingface/peft)
- Inspired by diffusion models for continuous domains and mask-based language modeling
- Dataset: [Tulu 3 SFT Mixture](https://huggingface.co/datasets/allenai/tulu-3-sft-mixture-0225) from AllenAI

## Related Work

- [Diffusion-LM](https://arxiv.org/abs/2205.14217) - Diffusion models for text generation
- [BERT](https://arxiv.org/abs/1810.04805) - Masked language modeling
- [LoRA](https://arxiv.org/abs/2106.09685) - Low-Rank Adaptation for efficient fine-tuning
- [QLoRA](https://arxiv.org/abs/2305.14314) - Efficient fine-tuning with quantization

---
## To run this do:

```bash
nohup ./run.sh > training_nohup.log 2>&1 &
```

to view the logs do:

```bash
tail -f training_nohup.log
```
