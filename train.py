import os
from dataclasses import dataclass
from typing import Optional, List
from datetime import datetime
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import torch.nn as nn
import torch
from accelerate.utils import set_seed, ProjectConfiguration
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoTokenizer, AutoModel, BitsAndBytesConfig
from transformers import get_cosine_schedule_with_warmup
from torch.utils.data import Dataset, DataLoader
from accelerate import Accelerator
import json
import random

@dataclass
class InputFeatures:
    query_input_ids: torch.Tensor
    query_attention_mask: torch.Tensor
    pos_input_ids: torch.Tensor
    pos_attention_mask: torch.Tensor
    neg_input_ids: List[torch.Tensor]
    neg_attention_mask: List[torch.Tensor]


class EmbeddingDataset(Dataset):
    def __init__(
        self,
        train_data_path: str,
        tokenizer,
        neg_nums: int = 2,
        query_max_len: int = 256,
        passage_max_len: int = 1024,
    ):
        self.tokenizer = tokenizer
        self.neg_nums = neg_nums
        self.query_max_len = query_max_len
        self.passage_max_len = passage_max_len
        self.data = []
        
        with open(train_data_path, 'r', encoding='utf-8') as f:
            for line in f:
                item = json.loads(line)
                self.data.append({
                    'query': item['query'],
                    'pos': item['pos'],
                    'neg': item['neg']
                })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        item = self.data[index]
        query = item['query']
        pos = item['pos'][0]  # Take first positive passage
        
        # neg = item['neg'][:self.neg_nums]  # Take specified number of negative passages
        neg = random.sample(item['neg'],self.neg_nums)
        query_encodings = self.tokenizer(
            query,
            max_length=self.query_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        pos_encodings = self.tokenizer(
            pos,
            max_length=self.passage_max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        neg_encodings_list = []
        neg_attention_mask_list = []
        for neg_text in neg:
            neg_encoding = self.tokenizer(
                neg_text,
                max_length=self.passage_max_len,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            neg_encodings_list.append(neg_encoding['input_ids'].squeeze(0))
            neg_attention_mask_list.append(neg_encoding['attention_mask'].squeeze(0))
        
        return InputFeatures(
            query_input_ids=query_encodings['input_ids'].squeeze(0),
            query_attention_mask=query_encodings['attention_mask'].squeeze(0),
            pos_input_ids=pos_encodings['input_ids'].squeeze(0),
            pos_attention_mask=pos_encodings['attention_mask'].squeeze(0),
            neg_input_ids=neg_encodings_list,
            neg_attention_mask=neg_attention_mask_list
        )

    def collate_fn(self, features):
        # batch_size = len(features)
        num_neg = len(features[0].neg_input_ids)
        
        # Stack queries and positive examples
        query_input_ids = torch.stack([f.query_input_ids for f in features])
        query_attention_mask = torch.stack([f.query_attention_mask for f in features])
        pos_input_ids = torch.stack([f.pos_input_ids for f in features])
        pos_attention_mask = torch.stack([f.pos_attention_mask for f in features])
        
        # Stack negative examples
        neg_input_ids = torch.stack([
            torch.stack([f.neg_input_ids[i] for f in features])
            for i in range(num_neg)
        ])
        neg_attention_mask = torch.stack([
            torch.stack([f.neg_attention_mask[i] for f in features])
            for i in range(num_neg)
        ])
        
        return InputFeatures(
            query_input_ids=query_input_ids,
            query_attention_mask=query_attention_mask,
            pos_input_ids=pos_input_ids,
            pos_attention_mask=pos_attention_mask,
            neg_input_ids=neg_input_ids,
            neg_attention_mask=neg_attention_mask
        )


class QLoRAEmbeddingModel(torch.nn.Module):
    def __init__(self, model, temperature=0.02):
        super().__init__()
        self.model = model
        self.temperature = temperature

    def mean_pooling(self, token_embeddings, attention_mask):
        bs, seq_len, hidden_dim = token_embeddings.shape
        values, indices = attention_mask.flip(1).max(1)
        indices = torch.where(values == 0, seq_len - 1, indices)
        gather_indices = seq_len - indices - 1

        # Turn indices from shape [bs] --> [bs, 1, hidden_dim]
        gather_indices = gather_indices.unsqueeze(-1).repeat(1, hidden_dim)
        gather_indices = gather_indices.unsqueeze(1)
        assert gather_indices.shape == (bs, 1, hidden_dim)

        # Gather along the 1st dim (seq_len) (bs, seq_len, hidden_dim -> bs, hidden_dim)
        # Actually no need for the attention mask as we gather the last token where attn_mask = 1
        # but as we set some indices (which shouldn't be attended to) to 0 with clamp, we
        # use the attention mask to ignore them again
        input_mask_expanded = (
            attention_mask.unsqueeze(-1).expand(token_embeddings.size()).to(token_embeddings.dtype)
        )
        embedding = torch.gather(token_embeddings * input_mask_expanded, 1, gather_indices).squeeze(dim=1)
        return embedding
        # input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        # return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode_passage(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.mean_pooling(outputs.last_hidden_state, attention_mask)
        return embeddings  # Removed normalization here as it's done in loss function

    def forward(self, query_input_ids, query_attention_mask, 
               pos_input_ids, pos_attention_mask,
               neg_input_ids, neg_attention_mask):
        
        # Get embeddings
        query_embeds = self.encode_passage(query_input_ids, query_attention_mask)
        pos_embeds = self.encode_passage(pos_input_ids, pos_attention_mask)
        
        # Get negative embeddings
        neg_embeds_list = []
        for i in range(len(neg_input_ids)):
            neg_embeds = self.encode_passage(neg_input_ids[i], neg_attention_mask[i])
            neg_embeds_list.append(neg_embeds)
        neg_embeds = torch.cat(neg_embeds_list, dim=0)

        # Normalize embeddings
        query_embeds = torch.nn.functional.normalize(query_embeds, p=2, dim=-1)
        pos_embeds = torch.nn.functional.normalize(pos_embeds, p=2, dim=-1)
        neg_embeds = torch.nn.functional.normalize(neg_embeds, p=2, dim=-1)

        # Calculate positive similarities
        pos_sim_matrix = torch.sum(query_embeds * pos_embeds, dim=-1)

        # Calculate negative similarities
        neg_sim_matrix = query_embeds.unsqueeze(1) @ neg_embeds.unsqueeze(0).transpose(-1, -2)
        neg_sim_matrix = neg_sim_matrix.squeeze(1)

        # Prepare labels (first position is positive)
        labels = torch.zeros(query_embeds.size(0), dtype=torch.long, device=query_embeds.device)

        # Combine positive and negative scores
        pos_neg_score = torch.cat([pos_sim_matrix.unsqueeze(1), neg_sim_matrix], dim=1) / self.temperature

        # Calculate loss
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(pos_neg_score, labels)
        # loss = torch.nn.functional.cross_entropy(pos_neg_score, labels)
        
        return loss


def create_adamw_optimizer(model, lr, weight_decay=1e-2):
    no_decay = ["bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in model.named_parameters() 
                      if not any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in model.named_parameters() 
                      if any(nd in n for nd in no_decay) and p.requires_grad],
            "weight_decay": 0.0,
        },
    ]
    return torch.optim.AdamW(optimizer_grouped_parameters, lr=lr)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", required=True)
    parser.add_argument("--train_dataset", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--neg_nums", type=int, default=2)
    parser.add_argument("--query_max_len", type=int, default=256)
    parser.add_argument("--passage_max_len", type=int, default=1024)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--save_on_epoch_end", type=int, default=1)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=16)
    parser.add_argument("--warmup_proportion", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log_with", type=str, default="tensorboard", 
                       help="Logging tool to use: tensorboard, wandb, or all")
    
    # Finetune type argument
    parser.add_argument("--finetune_type", type=str, choices=["qlora", "lora", "full"], 
                       default="qlora", help="Choose finetuning method: qlora, lora, or full")
    
    # Quantization argument for QLoRA
    parser.add_argument("--quantization_bits", type=int, choices=[4, 8], default=4,
                       help="Quantization bits for QLoRA (4 or 8 bit)")
    
    # QLoRA/LoRA specific arguments
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.1)
    
    args = parser.parse_args()
    # Ensure output_dir exists
    os.makedirs(args.output_dir, exist_ok=True)
    # Set seed for reproducibility
    set_seed(args.seed)
    
    # Initialize list to store epoch losses
    epoch_losses = []
    
    # Initialize detailed training logs
    training_logs = {
        "training_args": vars(args),
        "model_info": {},
        "step_logs": [],
        "epoch_logs": [],
        "training_start_time": None,
        "training_end_time": None
    }
    
    # Configure accelerator
    project_config = ProjectConfiguration(
        project_dir=os.path.join(args.output_dir, 'runs'),
        total_limit=5
    )
    
    # Adjust mixed precision based on finetune type
    mixed_precision = "bf16" if args.finetune_type in ["qlora", "lora"] else "no"
    
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        log_with=args.log_with,  # tensorboard, wandb, or all
        mixed_precision=mixed_precision,
        project_config=project_config
    )
    
    # Print logging information
    if args.log_with == "tensorboard":
        accelerator.print(f"TensorBoard logs will be saved to: {os.path.join(args.output_dir, 'runs')}")
        accelerator.print("To view logs, run: tensorboard --logdir=" + os.path.join(args.output_dir, 'runs'))
    
    # Initialize trackers
    config_dict = vars(args)
    config_dict["finetune_type"] = args.finetune_type
    
    # Set project name based on finetune type
    project_name = f"{args.finetune_type}_embedding_training"
    if args.finetune_type == "qlora":
        project_name += f"_{args.quantization_bits}bit"
    
    accelerator.init_trackers(project_name, config=config_dict)
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Model initialization based on finetune type
    if args.finetune_type == "qlora":
        # Configure quantization for QLoRA
        if args.quantization_bits == 4:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=False
            )
            accelerator.print(f"Using QLoRA training with 4-bit quantization")
        elif args.quantization_bits == 8:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_compute_dtype=torch.bfloat16
            )
            accelerator.print(f"Using QLoRA training with 8-bit quantization")
        else:
            raise ValueError(f"Unsupported quantization_bits: {args.quantization_bits}")
        
        # Load base model with quantization
        base_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare model for k-bit training
        base_model = prepare_model_for_kbit_training(base_model)
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["up_proj","q_proj","k_proj", "v_proj","gate_proj","o_proj","down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # Create LoRA model
        model = get_peft_model(base_model, lora_config)
        accelerator.print("Using QLoRA training")
        
    elif args.finetune_type == "lora":
        # Load base model without quantization
        base_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        
        # Configure LoRA
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=["up_proj","q_proj","k_proj", "v_proj","gate_proj","o_proj","down_proj"],
            lora_dropout=args.lora_dropout,
            bias="none",
            task_type="FEATURE_EXTRACTION"
        )
        
        # Create LoRA model
        model = get_peft_model(base_model, lora_config)
        accelerator.print("Using LoRA training")
        
    elif args.finetune_type == "full":
        # Load base model for full finetuning
        base_model = AutoModel.from_pretrained(
            args.model_name_or_path,
            torch_dtype=torch.float32,  # Use bf16 for full finetuning
            device_map="auto"
        )
        
        # For full finetuning, we use the model directly
        model = base_model
        accelerator.print("Using full finetuning")
        
    else:
        raise ValueError(f"Unsupported finetune_type: {args.finetune_type}")
    
    # Print trainable parameters info
    if args.finetune_type in ["qlora", "lora"]:
        model.print_trainable_parameters()
        # Record model info for logs
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        training_logs["model_info"] = {
            "finetune_type": args.finetune_type,
            "quantization_bits": args.quantization_bits if args.finetune_type == "qlora" else None,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "lora_config": {
                "r": args.lora_r,
                "alpha": args.lora_alpha,
                "dropout": args.lora_dropout
            } if args.finetune_type in ["qlora", "lora"] else None
        }
    else:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        accelerator.print(f"Total parameters: {total_params:,}")
        accelerator.print(f"Trainable parameters: {trainable_params:,}")
        accelerator.print(f"Trainable%: {100 * trainable_params / total_params:.2f}")
        # Record model info for logs
        training_logs["model_info"] = {
            "finetune_type": args.finetune_type,
            "quantization_bits": None,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "trainable_percentage": 100 * trainable_params / total_params,
            "lora_config": None
        }
    
    # Wrap model with embedding model class
    model = QLoRAEmbeddingModel(model, temperature=args.temperature)
    
    # Create dataset and dataloader
    train_dataset = EmbeddingDataset(
        train_data_path=args.train_dataset,
        tokenizer=tokenizer,
        neg_nums=args.neg_nums,
        query_max_len=args.query_max_len,
        passage_max_len=args.passage_max_len
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=train_dataset.collate_fn,
        shuffle=True,
        num_workers=16,
        pin_memory=True
    )
    
    # Create optimizer and scheduler
    optimizer = create_adamw_optimizer(model, lr=args.lr)
    num_training_steps = len(train_dataloader) * args.epochs
    num_warmup_steps = int(args.warmup_proportion * num_training_steps)
    lr_scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps
    )
    
    # Prepare for training
    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler
    )
    
    # Get device for tensor operations
    device = accelerator.device
    
    # Record training start time
    training_logs["training_start_time"] = datetime.now().isoformat()
    start_time = time.time()
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        epoch_start_time = time.time()
        
        for step, batch in enumerate(train_dataloader):
            step_start_time = time.time()
            
            with accelerator.accumulate(model):
                loss = model(
                    query_input_ids=batch.query_input_ids.to(device),
                    query_attention_mask=batch.query_attention_mask.to(device),
                    pos_input_ids=batch.pos_input_ids.to(device),
                    pos_attention_mask=batch.pos_attention_mask.to(device),
                    neg_input_ids=batch.neg_input_ids.to(device),
                    neg_attention_mask=batch.neg_attention_mask.to(device)
                )
                
                accelerator.backward(loss)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                
                total_loss += loss.detach().float()
            
            step_end_time = time.time()
            step_duration = step_end_time - step_start_time
            
            # Record detailed step information
            step_info = {
                "epoch": epoch + 1,
                "step": step + 1,
                "global_step": step + epoch * len(train_dataloader),
                "loss": float(loss.item()),
                "learning_rate": float(lr_scheduler.get_last_lr()[0]),
                "step_duration": step_duration,
                "timestamp": datetime.now().isoformat()
            }
            training_logs["step_logs"].append(step_info)
            
            accelerator.print(f"Epoch {epoch+1}/{args.epochs} | Step {step+1}/{len(train_dataloader)} | Loss: {loss.item():.4f} | LR: {lr_scheduler.get_last_lr()[0]:.2e}")
            accelerator.log({"train_loss": loss.item(), "learning_rate": lr_scheduler.get_last_lr()[0]}, 
                          step=step + epoch * len(train_dataloader))
        
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        avg_loss = total_loss / len(train_dataloader)
        
        accelerator.print(f"Epoch {epoch+1}/{args.epochs} | Average Loss: {avg_loss:.4f} | Duration: {epoch_duration:.2f}s")
        
        # Store detailed epoch information
        epoch_info = {
            "epoch": epoch + 1,
            "avg_loss": float(avg_loss),
            "total_loss": float(total_loss),
            "num_steps": len(train_dataloader),
            "epoch_duration": epoch_duration,
            "learning_rate": float(lr_scheduler.get_last_lr()[0]),
            "timestamp": datetime.now().isoformat()
        }
        training_logs["epoch_logs"].append(epoch_info)
        
        # Store epoch loss (for backward compatibility)
        epoch_losses.append({
            "epoch": epoch + 1,
            "loss": float(avg_loss)
        })
        
        # Save detailed training logs
        if accelerator.is_main_process:
            # Save step-by-step logs
            with open(os.path.join(args.output_dir, "detailed_training_logs.json"), 'w') as f:
                json.dump(training_logs, f, indent=2)
            
            # Save simple epoch losses (for backward compatibility)
            with open(os.path.join(args.output_dir, "training_losses.json"), 'w') as f:
                json.dump(epoch_losses, f, indent=2)
        
        if (epoch + 1) % args.save_on_epoch_end == 0:
            # Save the model based on finetune type
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            
            if args.finetune_type in ["qlora", "lora"]:
                # Save LoRA adapter only
                unwrapped_model.model.save_pretrained(
                    os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}"),
                    safe_serialization=True,
                    state_dict=accelerator.get_state_dict(model)
                )
            else:
                # Save full model
                unwrapped_model.model.save_pretrained(
                    os.path.join(args.output_dir, f"checkpoint-epoch-{epoch+1}"),
                    safe_serialization=True,
                    state_dict=accelerator.get_state_dict(model)
                )
    
    # Save the final model
    accelerator.wait_for_everyone()
    unwrapped_model = accelerator.unwrap_model(model)
    
    if args.finetune_type in ["qlora", "lora"]:
        # Save LoRA adapter
        unwrapped_model.model.save_pretrained(
            os.path.join(args.output_dir, "final"),
            safe_serialization=True,
            state_dict=accelerator.get_state_dict(model)
        )
    else:
        # Save full model
        unwrapped_model.model.save_pretrained(
            os.path.join(args.output_dir, "final"),
            safe_serialization=True,
            state_dict=accelerator.get_state_dict(model)
        )
    
    # Record training end time and summary
    training_logs["training_end_time"] = datetime.now().isoformat()
    total_training_time = time.time() - start_time
    training_logs["training_summary"] = {
        "total_training_time": total_training_time,
        "total_epochs": args.epochs,
        "total_steps": len(training_logs["step_logs"]),
        "final_loss": training_logs["epoch_logs"][-1]["avg_loss"] if training_logs["epoch_logs"] else None,
        "best_loss": min([epoch["avg_loss"] for epoch in training_logs["epoch_logs"]]) if training_logs["epoch_logs"] else None,
        "avg_step_duration": sum([step["step_duration"] for step in training_logs["step_logs"]]) / len(training_logs["step_logs"]) if training_logs["step_logs"] else None
    }
    
    # Save tokenizer
    if accelerator.is_main_process:
        tokenizer.save_pretrained(os.path.join(args.output_dir, "final"))
        
        # Save training configuration
        config_to_save = {
            "finetune_type": args.finetune_type,
            "model_name_or_path": args.model_name_or_path,
            "quantization_bits": args.quantization_bits if args.finetune_type == "qlora" else None,
            "lora_r": args.lora_r if args.finetune_type in ["qlora", "lora"] else None,
            "lora_alpha": args.lora_alpha if args.finetune_type in ["qlora", "lora"] else None,
            "lora_dropout": args.lora_dropout if args.finetune_type in ["qlora", "lora"] else None,
            "temperature": args.temperature,
            "final_epoch": args.epochs
        }
        
        with open(os.path.join(args.output_dir, "final", "training_config.json"), 'w') as f:
            json.dump(config_to_save, f, indent=2)
        
        # Save final detailed training logs
        with open(os.path.join(args.output_dir, "detailed_training_logs.json"), 'w') as f:
            json.dump(training_logs, f, indent=2)
        
        # Generate training summary report
        summary_report = {
            "training_completed": True,
            "completion_time": training_logs["training_end_time"],
            "total_duration_hours": total_training_time / 3600,
            "model_info": training_logs["model_info"],
            "training_args": training_logs["training_args"],
            "performance_summary": training_logs["training_summary"],
            "loss_progression": [{"epoch": e["epoch"], "avg_loss": e["avg_loss"]} for e in training_logs["epoch_logs"]]
        }
        
        with open(os.path.join(args.output_dir, "training_summary_report.json"), 'w') as f:
            json.dump(summary_report, f, indent=2)
        
        accelerator.print(f"Training completed! Total time: {total_training_time/3600:.2f} hours")
        accelerator.print(f"Final loss: {training_logs['training_summary']['final_loss']:.4f}")
        accelerator.print(f"Best loss: {training_logs['training_summary']['best_loss']:.4f}")
        accelerator.print(f"Average step duration: {training_logs['training_summary']['avg_step_duration']:.3f}s")
    
    accelerator.end_training()


if __name__ == "__main__":
    main()