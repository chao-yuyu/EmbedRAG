# EmbedRAG - Embedding Model Training Framework

A Transformer-based General Text Embedding model training framework that supports LoRA, QLoRA, and full parameter fine-tuning methods, specifically designed for Retrieval Augmented Generation (RAG) scenarios.

## Features

- 🚀 Support multiple fine-tuning strategies: LoRA, QLoRA (4-bit/8-bit), full parameter fine-tuning
- 📊 Detailed training logs and monitoring
- 🔧 Flexible parameter configuration
- 💾 Automatic model saving and checkpoint management
- 🔄 Seamless merging of LoRA adapters with base models

## System Requirements

- Python 3.10+
- CUDA GPU
- PyTorch
- Transformers
- PEFT
- Accelerate

## Install Dependencies

```bash
pip install torch transformers peft accelerate datasets safetensors bitsandbytes
```

## Training Data Format

Training data uses JSONL format, with each line containing a JSON object with the following fields:

```json
{
  "query": "Query text",
  "pos": ["Positive example 1", "Positive example 2"],
  "neg": ["Negative example 1", "Negative example 2", "Negative example 3"]
}
```

### Sample Data

```json
{"query": "Best beef noodle restaurant in Taipei", "pos": ["Lao Zhang Beef Noodle Restaurant is located in downtown Taipei, famous for its rich broth and tender beef, recommended by locals as a must-try delicacy"], "neg": ["Taipei 101 Observatory offers 360-degree city panoramic views", "Latest iPhone specifications comparison and price analysis", "Complete guide on how to grow vegetables at home"]}
{"query": "Machine learning beginner tutorial", "pos": ["Python Machine Learning Basics: Complete learning path from linear regression to deep learning, including practical examples and code"], "neg": ["Taipei night market food recommendation list", "2025 stock market investment strategy analysis", "Daily care considerations for pet dogs"]}
```

For more examples, please refer to `datasets/sample_data.jsonl`

## Training Commands

### 1. LoRA Training

Suitable for scenarios with limited GPU memory, high training efficiency:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name_or_path "/path/to/base/model" \
  --finetune_type lora \
  --train_dataset "/path/to/train_data.jsonl" \
  --output_dir "/path/to/output/lora_training" \
  --batch_size 1 \
  --lr 1e-4 \
  --epochs 3 \
  --neg_nums 4 \
  --temperature 0.02 \
  --query_max_len 128 \
  --passage_max_len 1000 \
  --save_on_epoch_end 1 \
  --warmup_proportion 0.05
```

### 2. QLoRA Training (4-bit quantization)

Further saves memory, suitable for large model fine-tuning:

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --model_name_or_path "/path/to/base/model" \
  --finetune_type qlora \
  --quantization_bits 4 \
  --train_dataset "/path/to/train_data.jsonl" \
  --output_dir "/path/to/output/qlora_training" \
  --batch_size 1 \
  --lr 1e-4 \
  --epochs 3 \
  --neg_nums 2 \
  --temperature 0.02 \
  --query_max_len 128 \
  --passage_max_len 1000 \
  --save_on_epoch_end 1 \
  --warmup_proportion 0.05
```

### 3. Full Parameter Fine-tuning

Suitable for environments with abundant resources, usually achieves the best results:

```bash
CUDA_VISIBLE_DEVICES=0,1 python train.py \
  --model_name_or_path "/path/to/base/model" \
  --finetune_type full \
  --train_dataset "/path/to/train_data.jsonl" \
  --output_dir "/path/to/output/full_training" \
  --batch_size 1 \
  --lr 1e-5 \
  --epochs 3 \
  --neg_nums 1 \
  --temperature 0.02 \
  --query_max_len 128 \
  --passage_max_len 1000 \
  --save_on_epoch_end 1 \
  --warmup_proportion 0.05
```

## Parameter Description

| Parameter | Description | Default |
|-----------|-------------|---------|
| `model_name_or_path` | Base model path | Required |
| `finetune_type` | Fine-tuning type: lora/qlora/full | qlora |
| `quantization_bits` | QLoRA quantization bits: 4/8 | 4 |
| `train_dataset` | Training data path | Required |
| `output_dir` | Output directory | Required |
| `batch_size` | Batch size | Required |
| `lr` | Learning rate | 1e-4 |
| `epochs` | Training epochs | 2 |
| `neg_nums` | Number of negative examples | 2 |
| `temperature` | Temperature parameter | 0.02 |
| `query_max_len` | Maximum query length | 256 |
| `passage_max_len` | Maximum document length | 1024 |
| `lora_r` | LoRA rank | 8 |
| `lora_alpha` | LoRA alpha | 32 |
| `lora_dropout` | LoRA dropout | 0.1 |

## Model Merging

### LoRA/QLoRA Model Merging

After training completion, you need to merge the LoRA adapter with the base model:

```bash
python merge.py \
    --base_model_path "/path/to/base/model" \
    --lora_adapter_path "/path/to/lora/adapter.safetensors" \
    --output_path "/path/to/merged/model" \
    --training_precision "float32"
```

### Merging Parameter Description

- `base_model_path`: Original base model path
- `lora_adapter_path`: LoRA adapter safetensors file path
- `output_path`: Output path for merged model
- `training_precision`: Precision used during training (float32/bfloat16)

## Training Monitoring

### TensorBoard Monitoring

```bash
tensorboard --logdir=/path/to/output/runs
```

### Training Logs

The training process generates the following log files:

- `detailed_training_logs.json`: Detailed step-by-step training logs
- `training_losses.json`: Average loss for each epoch
- `training_summary_report.json`: Training summary report

## Output Structure

Output directory structure after training completion:

```
output_dir/
├── final/                        # Final model
│   ├── adapter_model.safetensors # LoRA adapter (LoRA/QLoRA only)
│   ├── adapter_config.json       # LoRA configuration (LoRA/QLoRA only)  
│   ├── training_config.json      # Training configuration
│   └── tokenizer files...        # Tokenizer files
├── checkpoint-epoch-{N}/         # Checkpoint for each epoch
├── detailed_training_logs.json   # Detailed training logs
├── training_losses.json          # Loss records
├── training_summary_report.json  # Training summary
└── runs/                         # TensorBoard logs
```

## Usage Examples

### 1. Prepare Training Data

```python
# Create training data
import json

data = [
    {
        "query": "Your query",
        "pos": ["Relevant document 1", "Relevant document 2"],
        "neg": ["Irrelevant document 1", "Irrelevant document 2", "Irrelevant document 3"]
    }
]

with open("train_data.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

### 2. Execute Training

```bash
# LoRA training
python train.py \
  --model_name_or_path "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
  --finetune_type lora \
  --train_dataset "train_data.jsonl" \
  --output_dir "output/lora_training" \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 3
```

### 3. Merge Model

```bash
# Merge LoRA adapter
python merge.py \
    --base_model_path "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
    --lora_adapter_path "output/lora_training/final/adapter_model.safetensors" \
    --output_path "output/merged_model"
```

### 4. Model Evaluation Usage

> **Note**: For complete usage examples, please refer to the `evaluation.ipynb` file

```python
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# Define last token pooling function to extract embeddings from the last valid token of each sequence
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# Load model and tokenizer
model_path = "your/model/path"  # Replace with your model path
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="auto"  # Automatic GPU allocation
)

# Define queries and documents
queries = ["Your query text"]
documents = ["Document 1", "Document 2", "Document 3"]

# Combine queries and documents for batch processing
input_texts = queries + documents
batch_dict = tokenizer(
    input_texts, 
    max_length=512, 
    padding=True, 
    truncation=True, 
    return_tensors='pt'
)

# Move to GPU (if available)
if torch.cuda.is_available():
    batch_dict = {k: v.cuda() for k, v in batch_dict.items()}

# Get model output and extract embeddings
with torch.no_grad():
    outputs = model(**batch_dict, output_hidden_states=True)
    embeddings = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

# Calculate similarity scores
scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T) * 100

# Display results
for i, query in enumerate(queries):
    print(f"Query: {query}")
    query_scores = scores[i].tolist()
    for j, (doc, score) in enumerate(zip(documents, query_scores)):
        print(f"  Document{j+1}: {score:.2f} - {doc}")
```

## Important Notes

1. **Memory Management**: QLoRA requires less GPU memory, suitable for resource-constrained environments
2. **Learning Rate Adjustment**: Different fine-tuning methods recommend different learning rates (Full: 1e-5, LoRA/QLoRA: 1e-4)
3. **Number of Negative Examples**: Recommend adjusting neg_nums parameter based on GPU memory
4. **Precision Consistency**: Ensure training_precision is consistent with training when merging

## Troubleshooting

### Common Issues

1. **CUDA OOM**: Reduce batch_size or use QLoRA
2. **Merge Failure**: Check if LoRA configuration is consistent with training
3. **Precision Mismatch**: Ensure training and merging use the same precision settings

### Performance Optimization

- Use multiple GPUs: Specify multiple GPUs in CUDA_VISIBLE_DEVICES
- Adjust num_workers: Adjust DataLoader's num_workers based on CPU cores
- Use gradient accumulation: Increase gradient_accumulation_steps to simulate larger batch size

## Documentation

- [English README](README.md)
- [Chinese README](README_zh.md)

## License

MIT License 