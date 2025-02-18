import argparse
import os
import random
import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import load_dataset
import re

os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)
torch.manual_seed(42)
torch.cuda.empty_cache()

def extract_gsm8k_data(dataset, row_index):
    try:
        row = dataset[row_index]
        return {
            "prompt": row['question'],
            "completion": row['answer'],
            "ground_truth": row['answer'].split("#### ")[-1]
        }
    except IndexError:
        return None

def process_gsm8k(dataset):
    data = [result for i in range(len(dataset)) if (result := extract_gsm8k_data(dataset, i))]
    return Dataset.from_list(data)

def tokenize_fn(batch, tokenizer, max_length):
    prompt = tokenizer(
        batch["prompt"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    completion = tokenizer(
        batch["completion"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    ground_truth = tokenizer(
        batch["ground_truth"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    
    return {
        "prompt_input_ids": prompt["input_ids"],
        "prompt_attention_mask": prompt["attention_mask"],
        "completion_input_ids": completion["input_ids"],
        "completion_attention_mask": completion["attention_mask"],
        "ground_truth_input_ids": ground_truth["input_ids"],
        "ground_truth_attention_mask": ground_truth["attention_mask"],
    }

def reward_func(completions, ground_truth, **kwargs):
    matches = [re.search(r"\\boxed\{(.*?)\}", completion) for completion in completions]
    contents = [match.group(1) if match else "" for match in matches]
    return [1.0 if c == gt else 0.0 for c, gt in zip(contents, ground_truth)]


# def main(args):
#     device = torch.device("cuda" if torch.cuda.is_available() else "mps")
#     if args.report_to == "wandb":
#         wandb.init(project="grpo_training", config=args.__dict__)

#     train_data = load_dataset("openai/gsm8k", "main", split="train")
#     test_data = load_dataset("openai/gsm8k", "main", split="test")

#     tokenizer = AutoTokenizer.from_pretrained(args.model_name)
#     tokenizer.pad_token = tokenizer.eos_token

#     if args.num_rows > 0:
#         train_data = train_data.select(range(min(len(train_data), args.num_rows)))
#     if args.test_size > 0:
#         test_data = test_data.select(range(min(len(test_data), args.test_size)))

#     train_data = process_gsm8k(train_data)
#     test_data = process_gsm8k(test_data)

#     train_data = train_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)
#     test_data = test_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)

#     model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
#     ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
#     ref_model.eval()
#     for param in ref_model.parameters():
#         param.requires_grad = False

#     peft_config = LoraConfig(
#         task_type="CAUSAL_LM",
#         inference_mode=False,
#         r=args.lora_rank,
#         lora_alpha=16,
#         lora_dropout=0.1,
#     )

#     training_args = GRPOConfig(
#         output_dir=args.output_dir,
#         per_device_train_batch_size=args.per_device_train_batch_size,
#         per_device_eval_batch_size=args.per_device_eval_batch_size,
#         beta=args.beta,
#         logging_steps=args.logging_steps,
#         num_train_epochs=args.epochs,
#         gradient_accumulation_steps=args.gradient_accumulation_steps,
#         learning_rate=args.learning_rate,
#         eval_strategy="steps",
#         eval_steps=100,
#         fp16=torch.cuda.is_available(),
#         num_generations=2,
#         max_completion_length=512,
#         log_completions=True,
#     )

#     trainer = GRPOTrainer(
#         model=model,
#         reward_funcs=reward_func,
#         # ref_model=ref_model,
#         processing_class=tokenizer,
#         args=training_args,
#         train_dataset=train_data,
#         eval_dataset=test_data,
#         peft_config=peft_config,
#     )
#     trainer.train()

#     if args.report_to == "wandb":
#         wandb.finish()

# if __name__ == "__main__":
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    # parser.add_argument("--output_dir", type=str, default="./grpo_model")
    # parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    # parser.add_argument("--per_device_eval_batch_size", type=int, default=10)
    # parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    # parser.add_argument("--epochs", type=int, default=10)
    # parser.add_argument("--learning_rate", type=float, default=5e-5)
    # parser.add_argument("--max_length", type=int, default=1500)
    # parser.add_argument("--beta", type=float, default=0.01)
    # parser.add_argument("--lora_rank", type=int, default=12)
    # parser.add_argument("--num_rows", type=int, default=1000)
    # parser.add_argument("--test_size", type=int, default=100)
    # parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="wandb")
    # parser.add_argument("--logging_steps", type=int, default=100)
    # args = parser.parse_args()
    # main(args)

import argparse
import os
import random
import torch
import wandb
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer
from peft import LoraConfig
from datasets import load_dataset
import re
from torch.distributed import init_process_group
from torch.utils.data.distributed import DistributedSampler


def main(args):
    # Initialize distributed training
    if torch.cuda.device_count() > 1:
        init_process_group(backend='nccl')
        local_rank = int(os.environ.get("LOCAL_RANK", -1))
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    if args.report_to == "wandb" and (local_rank == -1 or local_rank == 0):
        wandb.init(project="grpo_training", config=args.__dict__)

    train_data = load_dataset("openai/gsm8k", "main", split="train")
    test_data = load_dataset("openai/gsm8k", "main", split="test")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.num_rows > 0:
        train_data = train_data.select(range(min(len(train_data), args.num_rows)))
    if args.test_size > 0:
        test_data = test_data.select(range(min(len(test_data), args.test_size)))

    train_data = process_gsm8k(train_data)
    test_data = process_gsm8k(test_data)

    train_data = train_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)
    test_data = test_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16,
        device_map="auto"  # This helps with automatic device assignment
    ).to(device)

    if torch.cuda.device_count() > 1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank
        )

    ref_model = AutoModelForCausalLM.from_pretrained(
        args.model_name, 
        torch_dtype=torch.float16,
        device_map="auto"
    ).to(device)
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    peft_config = LoraConfig(
        task_type="CAUSAL_LM",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
    )

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        beta=args.beta,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=100,
        fp16=torch.cuda.is_available(),
        num_generations=2,
        max_completion_length=512,
        log_completions=True,
        ddp_find_unused_parameters=False,  # Important for DDP
        local_rank=local_rank if torch.cuda.device_count() > 1 else -1,
    )

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_func,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=test_data,
        peft_config=peft_config,
    )
    trainer.train()

    if args.report_to == "wandb" and (local_rank == -1 or local_rank == 0):
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./grpo_model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lora_rank", type=int, default=12)
    parser.add_argument("--num_rows", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="wandb")
    parser.add_argument("--logging_steps", type=int, default=100)
    args = parser.parse_args()
    main(args)    
    # Launch with distributed training if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        torch.multiprocessing.spawn(main, args=(args,), nprocs=torch.cuda.device_count())
    else:
        main(args)