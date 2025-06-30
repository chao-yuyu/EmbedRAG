from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from safetensors.torch import load_file
import argparse
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_path", required=True, help="Path to base model")
    parser.add_argument("--lora_adapter_path", required=True, help="Path to LoRA adapter safetensors file")
    parser.add_argument("--output_path", required=True, help="Output path for merged model")
    parser.add_argument("--training_precision", choices=["float32", "bfloat16"], default="bfloat16", 
                       help="Precision used during training (should match training config)")
    
    args = parser.parse_args()
    
    # 根據訓練精度選擇加載精度
    torch_dtype = torch.bfloat16 if args.training_precision == "bfloat16" else torch.float32
    
    print(f"Loading base model with {args.training_precision} precision...")
    print(f"Base model path: {args.base_model_path}")
    
    # 載入基礎模型 - 使用與訓練時一致的精度
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,  # 關鍵：使用與訓練時一致的精度
        trust_remote_code=True
    )
    
    # 設置 LoRA 配置（需要與訓練時一致）
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["up_proj","q_proj","k_proj", "v_proj","gate_proj","o_proj","down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    # 創建 PEFT 模型
    model = get_peft_model(base_model, lora_config)
    print(f"Model loaded with dtype: {next(model.parameters()).dtype}")
    
    print(f"Loading LoRA weights from: {args.lora_adapter_path}")
    
    # 加載 LoRA 權重
    lora_weights = load_file(args.lora_adapter_path)
    new_state_dict = {}
    
    # 檢查 LoRA 權重的精度
    first_key = next(iter(lora_weights.keys()))
    lora_dtype = lora_weights[first_key].dtype
    print(f"LoRA weights dtype: {lora_dtype}")
    print(f"Base model dtype: {next(model.parameters()).dtype}")
    
    # 處理權重映射和精度轉換
    for key, value in lora_weights.items():
        # 統一格式
        new_key = key.replace("model.base_model.model.", "base_model.model.model.") \
                     .replace("lora_A.weight", "lora_A.default.weight") \
                     .replace("lora_B.weight", "lora_B.default.weight")
        
        if new_key in model.state_dict():
            # 確保 LoRA 權重與模型精度一致
            if value.dtype != torch_dtype:
                print(f"Converting {new_key} from {value.dtype} to {torch_dtype}")
                value = value.to(torch_dtype)
            new_state_dict[new_key] = value
        else:
            print(f"[Unmatched key] 原始: {key} → 嘗試轉換後: {new_key}")
    
    print(f"Successfully matched {len(new_state_dict)} LoRA parameters")
    
    # 載入 LoRA 權重
    print("Loading LoRA weights into model...")
    missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"Missing keys: {len(missing_keys)}")
        if len(missing_keys) < 10:  # 只印出少量 missing keys
            for key in missing_keys:
                print(f"  - {key}")
    
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) < 10:  # 只印出少量 unexpected keys
            for key in unexpected_keys:
                print(f"  - {key}")
    
    # 合併模型
    print("Merging LoRA weights with base model...")
    merge_model = model.merge_and_unload()
    
    print(f"Merged model dtype: {next(merge_model.parameters()).dtype}")
    
    # 確保輸出目錄存在
    os.makedirs(args.output_path, exist_ok=True)
    
    # 保存合併後的模型
    print(f"Saving merged model to: {args.output_path}")
    merge_model.save_pretrained(args.output_path)
    
    # 保存 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_path)
    
    # 保存合併配置信息
    merge_config = {
        "base_model_path": args.base_model_path,
        "lora_adapter_path": args.lora_adapter_path,
        "training_precision": args.training_precision,
        "final_dtype": str(next(merge_model.parameters()).dtype),
        "lora_config": {
            "r": 8,
            "lora_alpha": 32,
            "target_modules": ["up_proj","q_proj","k_proj", "v_proj","gate_proj","o_proj","down_proj"],
            "lora_dropout": 0.1
        }
    }
    
    import json
    with open(os.path.join(args.output_path, "merge_config.json"), 'w') as f:
        json.dump(merge_config, f, indent=2)
    
    print("Merge completed successfully!")
    print(f"Final model precision: {args.training_precision}")
    print(f"Output saved to: {args.output_path}")

if __name__ == "__main__":
    main() 