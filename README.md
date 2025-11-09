# CLIP-JEPA

A PyTorch implementation of CLIP-style contrastive learning using a Joint Embedding Predictive Architecture (JEPA) with dual encoders: a pretrained language model (Qwen3) for text and a pretrained vision model (MobileNetV4) for images. The model uses QLoRA for efficient fine-tuning.

## Overview

CLIP-JEPA combines the contrastive learning approach of CLIP with separate vision and language encoders to learn aligned embeddings between images and text. The architecture uses:
- **Text Encoder**: Qwen3-4B-Instruct-2507 (a large language model) with QLoRA adapters
- **Vision Encoder**: MobileNetV4 (configurable via timm) with trainable projection layers
- **Loss Functions**: Multiple contrastive loss variants including CyCLIP and SigLIP

## Features

- **Dual-Encoder Architecture**: Separate vision and text encoders with projection layers
- **Efficient Fine-tuning**: Uses QLoRA (4-bit quantization + LoRA) for memory-efficient training
- **Multiple Loss Functions**: Supports CLIP, CyCLIP, SigLIP, and CySigLIP losses
- **Hardware Flexibility**: Automatically detects and uses CUDA, Apple Silicon (MPS), or CPU
- **PyTorch Lightning Integration**: Streamlined training with distributed support
- **Weights & Biases Logging**: Built-in experiment tracking with similarity visualizations
- **Flexible Vision Models**: Supports any timm vision model

## Architecture

### Model Components

1. **Vision Encoder**
   - Base: Pretrained vision model from timm (default: `mobilenetv4_hybrid_medium.ix_e550_r256_in1k`)
   - Projection: Multi-layer projection network with residual connections and layer normalization
   - Output: L2-normalized embeddings (512-dim by default)

2. **Text Encoder**
   - Base: Qwen3-4B-Instruct-2507 (4-bit quantized with QLoRA)
   - Special tokens: `<EMBED>` and `</EMBED>` to mark embedding extraction positions
   - Delta embeddings: Learnable deltas added to special token embeddings
   - Projection: Multi-layer projection network matching vision encoder
   - Output: L2-normalized embeddings (512-dim by default)

3. **Training Strategy**
   - Vision base model: Fine-tuned with 0.1x learning rate
   - Vision projection: Full learning rate
   - Text model: QLoRA adapters + delta embeddings (full learning rate)
   - Text projection: Full learning rate
   - Loss temperature: Learnable parameter with sigmoid activation

### Embedding Extraction Process

**Text Embedding:**
1. Text is formatted using the Qwen chat template
2. `</EMBED>` token is appended to mark embedding position
3. Text is processed through the LLM with QLoRA adapters
4. Hidden state at `</EMBED>` token position is extracted
5. Extracted features are projected and L2-normalized

**Image Embedding:**
1. Images are preprocessed using timm transforms
2. Features are extracted using the vision model
3. Features are projected and L2-normalized

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

## Usage

### Basic Training

Use the VS Code debugger with the `Train` configuration (configured in `.vscode/launch.json`).

Alternatively, run from command line:

```bash
python -m src.main
```

### Custom Hyperparameters

Pass hyperparameters as a JSON string:

```bash
python -m src.main --hyper-parameters-json '{
  "epochs": 10,
  "batch_size": 16,
  "learning_rate": 1e-3,
  "loss_type": "cyclip_sigmoid",
  "debug": false
}'
```

### Using config.yaml

You can also modify `config.yaml` and load it programmatically (though the current main.py expects JSON):

```yaml
debug: false
batch_size: 64
epochs: 2
learning_rate: 1.0e-3
log_every_n_steps: 50

lora_config:
  use_qlora: true

data_config:
  dataset: "jxie/coco_captions"
```

### Configuration Options

All hyperparameters can be configured via JSON. Key options include:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `epochs` | 5 | Number of training epochs |
| `batch_size` | 8 | Training batch size |
| `learning_rate` | 5e-4 | Base learning rate |
| `loss_type` | `"cyclip_sigmoid"` | Loss function (clip, cyclip, sigmoid, cyclip_sigmoid) |
| `temperature` | -1.0 | Initial logit temperature (sigmoid applied) |
| `accumulate_grad_batches` | 1 | Gradient accumulation steps |
| `lr_scheduler` | true | Use OneCycleLR scheduler |
| `debug` | false | Debug mode (limits batches, shows progress bar) |

See `src/config.py` for all available configuration options.

## Loss Functions

### CLIP Loss (`clip`)
Standard contrastive loss with symmetric image-to-text and text-to-image objectives using softmax normalization.

```
L = 0.5 * (L_i2t + L_t2i)
```

### CyCLIP Loss (`cyclip`)
Adds cycle-consistency regularization to CLIP loss:
- **Symmetry loss**: Encourages similarity matrix S to equal S^T
- **Modality difference loss**: Aligns within-modality similarity matrices

```
L = 0.5 * (L_i2t + L_t2i) + λ₁ * MSE(S, S^T) + λ₂ * MSE(I@I^T, T@T^T)
```

### SigLIP Loss (`sigmoid`)
Uses sigmoid-based binary cross-entropy instead of softmax, treating each pair independently.

```
L = BCE_with_logits(S / τ, eye(N))
```

### CySigLIP Loss (`cyclip_sigmoid`)
Combines sigmoid loss with cycle-consistency regularization.

```
L = SigLIP_loss + λ₁ * MSE(S, S^T) + λ₂ * MSE(I@I^T, T@T^T)
```

All losses include a learnable temperature parameter τ with sigmoid activation.

## Model Configuration

### Vision Model Settings

```python
vision_model_config:
  vision_model: "mobilenetv4_hybrid_medium.ix_e550_r256_in1k"  # Any timm model
  projection_layers: 3
  embed_dims: 512
```

Supported vision models: Any model from `timm` (PyTorch Image Models). Examples:
- `mobilenetv4_hybrid_medium.ix_e550_r256_in1k` (default, MPS compatible)
- `resnet50.a1_in1k`
- `vit_base_patch16_224.augreg_in21k`
- `efficientnet_b0.ra_in1k`

### Language Model Settings

```python
llm_model_config:
  model_name: "Qwen/Qwen3-4B-Instruct-2507"
  max_pixels: 147456  # 384 x 384
  embed_start_token: "<EMBED>"
  embed_end_token: "</EMBED>"
  max_length: 1024
  projection_layers: 3
  embed_dims: 512
```

### QLoRA Settings

```python
lora_config:
  use_qlora: true  # 4-bit quantization
  use_dora: false  # DoRA doesn't work with quantized models
  lora_rank: 8
  lora_alpha: 16
  lora_dropout: 0.05
  target_modules: ["qkv", "fc1", "fc2", "linear", "q_proj", "k_proj", "v_proj", 
                   "o_proj", "gate_proj", "up_proj", "down_proj"]
  modules_to_save: []  # delta_on_embedding is added automatically
```

## Dataset

By default, uses the `sayakpaul/coco-30-val-2014` dataset. Configure in hyperparameters:

```json
{
  "data_config": {
    "dataset": "jxie/coco_captions",
    "test_size": 0.1,
    "num_workers": 4,
    "pin_memory": true
  }
}
```

Dataset should be a Hugging Face dataset with:
- An `image` column (PIL Image or path)
- A text column with captions (auto-detected: `caption`, `text`, `sentence`, etc.)

## Training Outputs

Training artifacts are saved to:
- **Model checkpoints**: `/tmp/output/` (configurable via `output_dir`)
- **LoRA adapters**: Saved automatically on checkpoint (see `CLIPJepaTrainer.on_save_checkpoint`)
- **Lightning logs**: `lightning_logs/` directory
- **Wandb logs**: `/tmp/wandb/` (configurable via `wandb_log_path`)

## Project Structure

```
clip_jepa/
├── src/
│   ├── config.py         # Configuration classes and hyperparameters
│   ├── data.py           # Dataset loading and dataloaders
│   ├── losses.py         # Loss function implementations (CLIP, CyCLIP, SigLIP, CySigLIP)
│   ├── main.py           # Training entry point
│   ├── metrics.py        # Evaluation metrics (top-k accuracy)
│   ├── model.py          # Text encoder and projection layers
│   ├── trainer.py        # PyTorch Lightning trainer
│   └── vision_model.py   # Vision encoder and projection layers
├── notebooks/
│   ├── data_testing.ipynb
│   └── qwen_testing.ipynb
├── .vscode/
│   └── launch.json       # VS Code debug configurations
├── config.yaml           # Example YAML configuration
├── requirements.txt      # Python dependencies
├── setup.sh             # Environment setup script
└── README.md
```

## Requirements

- Python 3.10+
- PyTorch 2.8.0+
- transformers (Hugging Face)
- peft (LoRA/QLoRA)
- lightning (PyTorch Lightning)
- datasets (Hugging Face)
- timm (PyTorch Image Models)
- wandb
- bitsandbytes (for QLoRA quantization)
- And more (see `requirements.txt`)

## Hardware Requirements

- **Minimum**: CPU (very slow training, not recommended)
- **Recommended**: NVIDIA GPU with 16GB+ VRAM (uses 4-bit quantization for efficiency)
- **Apple Silicon**: MPS backend supported for M1/M2/M3 Macs

The model automatically selects the best available device. With QLoRA (4-bit quantization), the Qwen3-4B model requires significantly less memory than full precision training.

## Monitoring

Training metrics are logged to Weights & Biases:

- **Losses**: Training and validation loss per step/epoch
- **Accuracies**: Image→Text and Text→Image top-1 accuracy
- **Temperature**: Learnable temperature parameter value
- **Visualizations**: Similarity matrices with matched image-text pairs (first validation batch each epoch)

Configure W&B settings:

```python
wandb_config:
  project: "clip-jepa"
  entity: "your-wandb-username"
  wandb_log_path: "/tmp/wandb"
```

Set `debug: true` to prefix runs with "debug-" for easier filtering.

## Development

### Testing Components

```bash
# Test data loading
python -m src.data

# Run in debug mode (limited batches, progress bar)
python -m src.main --hyper-parameters-json '{"debug": true}'
```

### Notebooks

Explore the notebooks for experimentation:
- `notebooks/data_testing.ipynb` - Dataset exploration and visualization
- `notebooks/qwen_testing.ipynb` - Model testing and inference

### Debug Configuration

The `.vscode/launch.json` includes a `Train` configuration for easy debugging with VS Code.

## Advanced Usage

### Using Different Vision Models

```bash
python -m src.main --hyper-parameters-json '{
  "vision_model_config": {
    "vision_model": "vit_base_patch16_224.augreg_in21k",
    "projection_layers": 3,
    "embed_dims": 512
  }
}'
```

### Adjusting Learning Rates

The trainer uses different learning rates for different components:
- Vision base: `learning_rate / 10` (fine-tuning pretrained features)
- Vision projection: `learning_rate` (training from scratch)
- LLM QLoRA adapters: `learning_rate` (adapters are small)
- LLM projection: `learning_rate` (training from scratch)
- Loss parameters (temperature): `learning_rate / 10` (stability)

### Using Regular LoRA (No Quantization)

```bash
python -m src.main --hyper-parameters-json '{
  "lora_config": {
    "use_qlora": false,
    "use_dora": true,
    "lora_rank": 32
  }
}'
```

Note: This will require significantly more memory.

## Technical Details

### Delta Embeddings

The `DeltaOnEmbedding` module adds learnable deltas to the `<EMBED>` and `</EMBED>` token embeddings without modifying the base embedding table. This allows the model to learn special representations for these tokens while keeping the pretrained embeddings frozen.

### Projection Networks

Both vision and text encoders use multi-layer projection networks with:
- Residual connections
- SiLU activations
- Layer normalization
- Dropout regularization
- Final L2 normalization for contrastive learning

### Mixed Precision Training

The trainer automatically uses bfloat16 precision on CUDA and MPS devices for faster training and lower memory usage.

## Troubleshooting

### CUDA Out of Memory

- Reduce `batch_size`
- Increase `accumulate_grad_batches`
- Ensure `use_qlora: true` for 4-bit quantization
- Use a smaller vision model

### MPS (Apple Silicon) Issues

- The default MobileNetV4 model is chosen for MPS compatibility
- Some operations may fall back to CPU automatically
- Use `debug: true` to test with small batches first

### W&B Login Required

```bash
wandb login
```

Or set environment variable:
```bash
export WANDB_API_KEY=your_api_key
```

## License

[Add your license here]

## Citation

If you use this code in your research, please cite:

```bibtex
[Add citation if applicable]
```

## Acknowledgments

- Built on [Qwen3](https://huggingface.co/Qwen) from Alibaba Cloud
- Vision models from [timm](https://github.com/huggingface/pytorch-image-models) (PyTorch Image Models)
- Inspired by:
  - [CLIP](https://arxiv.org/abs/2103.00020) (OpenAI)
  - [CyCLIP](https://arxiv.org/abs/2205.02459) (Cycle-Consistent CLIP)
  - [SigLIP](https://arxiv.org/abs/2303.15343) (Sigmoid Loss for Language Image Pre-training)
- Uses LoRA/QLoRA implementation from [PEFT](https://github.com/huggingface/peft)
- Training framework: [PyTorch Lightning](https://lightning.ai/)
