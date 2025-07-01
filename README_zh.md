# EmbedRAG - Embedding Model Training Framework

一個基於Transformer的General Text Embedding模型訓練框架，支持LoRA、QLoRA和全參數微調三種訓練方式，專為檢索增強生成(RAG)場景設計。

## 功能特色

- 🚀 支持多種微調策略：LoRA、QLoRA(4-bit/8-bit)、全參數微調
- 📊 詳細的訓練日誌和監控
- 🔧 靈活的參數配置
- 💾 自動模型保存和檢查點管理
- 🔄 LoRA adapter與base model的無縫合併

## 系統需求

- Python 3.10+
- CUDA GPU
- PyTorch
- Transformers
- PEFT
- Accelerate

## 安裝依賴

```bash
pip install torch transformers peft accelerate datasets safetensors bitsandbytes
```

## 訓練資料格式

訓練資料使用JSONL格式，每行包含一個JSON對象，包含以下字段：

```json
{
  "query": "查詢文本",
  "pos": ["正例文本1", "正例文本2"],
  "neg": ["負例文本1", "負例文本2", "負例文本3"]
}
```

### 範例資料

```json
{"query": "台北最好的牛肉麵店", "pos": ["老張牛肉麵館位於台北市中心，以其濃郁湯頭和嫩滑牛肉聞名，是當地人推薦的必吃美食"], "neg": ["台北101觀景台提供360度城市全景", "最新iPhone手機規格比較和價格分析", "如何在家種植蔬菜的完整指南"]}
{"query": "機器學習入門教程", "pos": ["Python機器學習基礎：從線性回歸到深度學習的完整學習路徑，包含實作範例和代碼"], "neg": ["台北夜市美食推薦清單", "2025年股市投資策略分析", "寵物狗的日常照護注意事項"]}
```

更多範例請參考 `datasets/sample_data.jsonl`

## 訓練指令

### 1. LoRA 訓練

適合有限GPU記憶體的場景，訓練效率高：

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

### 2. QLoRA 訓練 (4-bit量化)

進一步節省記憶體，適合大模型微調：

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

### 3. 全參數微調

適合充足資源環境，效果通常最佳：

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

## 參數說明

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `model_name_or_path` | 基礎模型路徑 | 必填 |
| `finetune_type` | 微調類型: lora/qlora/full | qlora |
| `quantization_bits` | QLoRA量化位數: 4/8 | 4 |
| `train_dataset` | 訓練資料路徑 | 必填 |
| `output_dir` | 輸出目錄 | 必填 |
| `batch_size` | 批次大小 | 必填 |
| `lr` | 學習率 | 1e-4 |
| `epochs` | 訓練輪數 | 2 |
| `neg_nums` | 負例數量 | 2 |
| `temperature` | 溫度參數 | 0.02 |
| `query_max_len` | 查詢最大長度 | 256 |
| `passage_max_len` | 文檔最大長度 | 1024 |
| `lora_r` | LoRA rank | 8 |
| `lora_alpha` | LoRA alpha | 32 |
| `lora_dropout` | LoRA dropout | 0.1 |

## 模型合併

### LoRA/QLoRA 模型合併

訓練完成後，需要將LoRA adapter與base model合併：

```bash
python merge.py \
    --base_model_path "/path/to/base/model" \
    --lora_adapter_path "/path/to/lora/adapter.safetensors" \
    --output_path "/path/to/merged/model" \
    --training_precision "float32"
```

### 合併參數說明

- `base_model_path`: 原始基礎模型路徑
- `lora_adapter_path`: LoRA adapter的safetensors檔案路徑
- `output_path`: 合併後模型的輸出路徑
- `training_precision`: 訓練時使用的精度 (float32/bfloat16)

## 訓練監控

### TensorBoard 監控

```bash
tensorboard --logdir=/path/to/output/runs
```

### 訓練日誌

訓練過程會生成以下日誌檔案：

- `detailed_training_logs.json`: 詳細的step-by-step訓練日誌
- `training_losses.json`: 每個epoch的平均loss
- `training_summary_report.json`: 訓練總結報告

## 輸出結構

訓練完成後的輸出目錄結構：

```
output_dir/
├── final/                        # 最終模型
│   ├── adapter_model.safetensors # LoRA adapter (僅LoRA/QLoRA)
│   ├── adapter_config.json       # LoRA配置 (僅LoRA/QLoRA)  
│   ├── training_config.json      # 訓練配置
│   └── tokenizer files...        # Tokenizer檔案
├── checkpoint-epoch-{N}/         # 各epoch檢查點
├── detailed_training_logs.json   # 詳細訓練日誌
├── training_losses.json          # Loss記錄
├── training_summary_report.json  # 訓練總結
└── runs/                         # TensorBoard日誌
```

## 使用範例

### 1. 準備訓練資料

```python
# 建立訓練資料
import json

data = [
    {
        "query": "你的查詢",
        "pos": ["相關文檔1", "相關文檔2"],
        "neg": ["不相關文檔1", "不相關文檔2", "不相關文檔3"]
    }
]

with open("train_data.jsonl", "w", encoding="utf-8") as f:
    for item in data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

### 2. 執行訓練

```bash
# LoRA訓練
python train.py \
  --model_name_or_path "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
  --finetune_type lora \
  --train_dataset "train_data.jsonl" \
  --output_dir "output/lora_training" \
  --batch_size 2 \
  --lr 1e-4 \
  --epochs 3
```

### 3. 合併模型

```bash
# 合併LoRA adapter
python merge.py \
    --base_model_path "Alibaba-NLP/gte-Qwen2-1.5B-instruct" \
    --lora_adapter_path "output/lora_training/final/adapter_model.safetensors" \
    --output_path "output/merged_model"
```

### 4. 使用模型評估

> **注意**: 完整的使用範例請參考 `evaluation.ipynb` 文件

```python
import torch
import torch.nn.functional as F
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

# 定義last token pooling函數，提取每個序列中最後一個有效token的embedding
def last_token_pool(last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

# 載入模型和tokenizer
model_path = "your/model/path"  # 替換為你的模型路徑
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModel.from_pretrained(
    model_path,
    torch_dtype=torch.float32,
    trust_remote_code=True,
    device_map="auto"  # 自動分配GPU
)

# 定義查詢和文檔
queries = ["你的查詢文本"]
documents = ["文檔1", "文檔2", "文檔3"]

# 合併查詢和文檔進行批次處理
input_texts = queries + documents
batch_dict = tokenizer(
    input_texts, 
    max_length=512, 
    padding=True, 
    truncation=True, 
    return_tensors='pt'
)

# 移動到GPU（如果可用）
if torch.cuda.is_available():
    batch_dict = {k: v.cuda() for k, v in batch_dict.items()}

# 獲取模型輸出並提取embeddings
with torch.no_grad():
    outputs = model(**batch_dict, output_hidden_states=True)
    embeddings = last_token_pool(outputs.hidden_states[-1], batch_dict['attention_mask'])
    embeddings = F.normalize(embeddings, p=2, dim=1)

# 計算相似度分數
scores = (embeddings[:len(queries)] @ embeddings[len(queries):].T) * 100

# 顯示結果
for i, query in enumerate(queries):
    print(f"查詢: {query}")
    query_scores = scores[i].tolist()
    for j, (doc, score) in enumerate(zip(documents, query_scores)):
        print(f"  文檔{j+1}: {score:.2f} - {doc}")
```

## 注意事項

1. **記憶體管理**: QLoRA需要較少GPU記憶體，適合資源受限環境
2. **學習率調整**: 不同微調方式建議使用不同學習率 (Full: 1e-5, LoRA/QLoRA: 1e-4)
3. **負例數量**: 建議根據GPU記憶體調整neg_nums參數
4. **精度一致性**: 合併時確保training_precision與訓練時一致

## 故障排除

### 常見問題

1. **CUDA OOM**: 減少batch_size或使用QLoRA
2. **合併失敗**: 檢查LoRA配置是否與訓練時一致
3. **精度不匹配**: 確保訓練和合併使用相同的精度設定

### 效能優化

- 使用多GPU: 在CUDA_VISIBLE_DEVICES中指定多個GPU
- 調整num_workers: 根據CPU核心數調整DataLoader的num_workers
- 使用梯度累積: 增加gradient_accumulation_steps以模擬更大的batch size

## License

MIT License 