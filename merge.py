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
    parser.add_argument("--training_precision", choices=["float32", "bfloat16"], default="float32", 
                       help="Precision used during training (should match training config)")
    
    args = parser.parse_args()
    
    # Choose loading precision based on training precision
    torch_dtype = torch.bfloat16 if args.training_precision == "bfloat16" else torch.float32
    
    print(f"Loading base model with {args.training_precision} precision...")
    print(f"Base model path: {args.base_model_path}")
    
    # Load base model - use precision consistent with training
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model_path,
        torch_dtype=torch_dtype,  # Key: use precision consistent with training
        trust_remote_code=True
    )
    
    # Set LoRA configuration (must be consistent with training)
    lora_config = LoraConfig(
        r=8,
        lora_alpha=32,
        target_modules=["up_proj","q_proj","k_proj", "v_proj","gate_proj","o_proj","down_proj"],
        lora_dropout=0.1,
        bias="none",
        task_type="FEATURE_EXTRACTION"
    )
    
    # Create PEFT model
    model = get_peft_model(base_model, lora_config)
    print(f"Model loaded with dtype: {next(model.parameters()).dtype}")
    
    print(f"Loading LoRA weights from: {args.lora_adapter_path}")
    
    # Load LoRA weights
    lora_weights = load_file(args.lora_adapter_path)
    new_state_dict = {}
    
    # Check LoRA weights precision
    first_key = next(iter(lora_weights.keys()))
    lora_dtype = lora_weights[first_key].dtype
    print(f"LoRA weights dtype: {lora_dtype}")
    print(f"Base model dtype: {next(model.parameters()).dtype}")
    
    # Handle weight mapping and precision conversion
    for key, value in lora_weights.items():
        # Unify format
        new_key = key.replace("model.base_model.model.", "base_model.model.model.") \
                     .replace("lora_A.weight", "lora_A.default.weight") \
                     .replace("lora_B.weight", "lora_B.default.weight")
        
        if new_key in model.state_dict():
            # Ensure LoRA weights match model precision
            if value.dtype != torch_dtype:
                print(f"Converting {new_key} from {value.dtype} to {torch_dtype}")
                value = value.to(torch_dtype)
            new_state_dict[new_key] = value
        else:
            print(f"[Unmatched key] Original: {key} → After conversion attempt: {new_key}")
    
    print(f"Successfully matched {len(new_state_dict)} LoRA parameters")
    
    # Load LoRA weights
    print("Loading LoRA weights into model...")
    merging_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
    
    if merging_keys:
        print(f"Merging keys: {len(merging_keys)}")
        if len(merging_keys) < 10:  # Only print a few merging_keys
            for key in merging_keys:
                print(f"  - {key}")
    
    if unexpected_keys:
        print(f"Unexpected keys: {len(unexpected_keys)}")
        if len(unexpected_keys) < 10:  # Only print a few unexpected keys
            for key in unexpected_keys:
                print(f"  - {key}")
    
    # Merge model
    print("Merging LoRA weights with base model...")
    merge_model = model.merge_and_unload()
    
    print(f"Merged model dtype: {next(merge_model.parameters()).dtype}")
    
    # Ensure output directory exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # Save merged model
    print(f"Saving merged model to: {args.output_path}")
    merge_model.save_pretrained(args.output_path)
    
    # Save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_path)
    tokenizer.save_pretrained(args.output_path)
    
    # Save merge configuration information
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