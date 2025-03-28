import argparse
import os
import json
import random
import torch
import wandb
import gzip
import shutil
import requests
import pandas as pd
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from trl import RewardTrainer, RewardConfig
from peft import LoraConfig
from datasets import concatenate_datasets
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache()

def download_and_decompress(url, output_file):
    compressed_file = output_file + ".gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open(compressed_file, "wb") as f:
            f.write(response.content)
        with gzip.open(compressed_file, "rb") as gz, open(output_file, "wb") as out:
            shutil.copyfileobj(gz, out)
    else:
        raise Exception(f"Failed to download {compressed_file}: HTTP {response.status_code}")

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def process_conversation(text):
    lines = text.strip().splitlines()
    messages = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        if ":" in line:
            role, content = line.split(":", 1)
            if role.strip() == "Human":
                role = "user"
            elif role.strip() == "Assistant":
                role = "assistant"
            messages.append({"role": role.strip(), "content": content.strip()})
    if not messages or messages[0]["role"] != "user":
        return []
    for i in range(1, len(messages)):
        expected_role = "assistant" if i % 2 == 1 else "user"
        if messages[i]["role"] != expected_role:
            return []
    return messages

def process_default(dataset):
    data_list = dataset.to_list()
    valid_rows = []
    for text in data_list:
        chosen = process_conversation(text["chosen"])
        rejected = process_conversation(text["rejected"])
        if chosen and rejected:
            valid_rows.append({"chosen": chosen, "rejected": rejected, "dID": text["dID"]})
    df = pd.DataFrame(valid_rows)
    df["row_id"] = df.index.values
    return Dataset.from_pandas(df)

def sample_and_shuffle(data_helpful, data_harmless, num_samples, weight):
    if num_samples > 0:
        n_helpful = int(weight * num_samples)
        n_harmless = num_samples - n_helpful
        n_helpful = min(n_helpful, len(data_helpful))
        n_harmless = min(n_harmless, len(data_harmless))
        sampled_helpful = data_helpful.select(random.sample(range(len(data_helpful)), n_helpful))
        sampled_harmless = data_harmless.select(random.sample(range(len(data_harmless)), n_harmless))
        sampled_helpful = [{"dID": 0, **entry} for entry in sampled_helpful.to_list()]
        sampled_harmless = [{"dID": 1, **entry} for entry in sampled_harmless.to_list()]
        combined_data = sampled_helpful + sampled_harmless
        random.shuffle(combined_data)
        return Dataset.from_list(combined_data)
    return Dataset.from_list([])

def process_batch(batch, tokenizer, args):
    chosen_texts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in batch["chosen"]]
    rejected_texts = [tokenizer.apply_chat_template(x, tokenize=False, add_generation_prompt=True) for x in batch["rejected"]]
    chosen_encodings = tokenizer(chosen_texts, max_length=args.max_length, truncation=True, padding="max_length")
    rejected_encodings = tokenizer(rejected_texts, max_length=args.max_length, truncation=True, padding="max_length")
    batch["input_ids_chosen"] = chosen_encodings["input_ids"]
    batch["attention_mask_chosen"] = chosen_encodings["attention_mask"]
    batch["input_ids_rejected"] = rejected_encodings["input_ids"]
    batch["attention_mask_rejected"] = rejected_encodings["attention_mask"]
    return batch

def main(args):
    print("using", torch.cuda.device_count(), "GPUs!")
    base_output_dir = args.output_dir
    unique_output_dir = os.path.join(base_output_dir, f"{args.model_name.replace('/', '_')}_seed{args.seed}")
    os.makedirs(unique_output_dir, exist_ok=True)
    print(f"Saving outputs to: {unique_output_dir}")
    if torch.cuda.is_available():
        device = torch.device("cuda")
        use_fp16 = False
        use_bf16 = True
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
        use_fp16 = False
    else:
        device = torch.device("cpu")
        use_fp16 = False
    print(f"Using device: {device}, FP16: {use_fp16}")
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    np.random.seed(args.seed)
    if args.report_to == "wandb":
        wandb.init(project="reward_group_training", config=args.__dict__)
    base_url = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/"
    for file in ["train.jsonl.gz", "test.jsonl.gz"]:
        download_and_decompress(base_url + file, file.replace(".gz", ""))
    train_data_helpful = Dataset.from_list(load_jsonl("train.jsonl"))
    test_data_helpful = Dataset.from_list(load_jsonl("test.jsonl"))
    base_url = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/"
    for file in ["train.jsonl.gz", "test.jsonl.gz"]:
        download_and_decompress(base_url + file, file.replace(".gz", ""))
    train_data_harmless = Dataset.from_list(load_jsonl("train.jsonl"))
    test_data_harmless = Dataset.from_list(load_jsonl("test.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    train_data = sample_and_shuffle(train_data_helpful, train_data_harmless, args.num_rows, args.weight)
    test_data_helpful = sample_and_shuffle(test_data_helpful, Dataset.from_list([]), args.test_size, 1)
    test_data_harmless = sample_and_shuffle(Dataset.from_list([]), test_data_harmless, args.test_size, 0)
    train_data = process_default(train_data)
    test_data_helpful = process_default(test_data_helpful)
    test_data_harmless = process_default(test_data_harmless)
    train_data = train_data.map(process_batch, batched=True, fn_kwargs={"tokenizer": tokenizer, "args": args})
    test_data_helpful = test_data_helpful.map(process_batch, batched=True, fn_kwargs={"tokenizer": tokenizer, "args": args})
    test_data_harmless = test_data_harmless.map(process_batch, batched=True, fn_kwargs={"tokenizer": tokenizer, "args": args})
    test_data = concatenate_datasets([test_data_harmless, test_data_helpful])
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=1,
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map="auto",
    )
    model.config.pad_token_id = tokenizer.pad_token_id
    peft_config = LoraConfig(
        task_type="SEQ_CLS",
        inference_mode=False,
        r=args.lora_rank,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    arg_remove_unused_columns = False
    training_args = RewardConfig(
        output_dir=unique_output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_length=args.max_length,
        remove_unused_columns=arg_remove_unused_columns,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=args.logging_steps,
        optim="adamw_torch",
        report_to=args.report_to,
        fp16=use_fp16,
        bf16=use_bf16,
        max_grad_norm=1.0,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=5,
        load_best_model_at_end=True,
        greater_is_better=True,
        seed = args.seed,
        data_seed = args.seed,
    )
    trainer = RewardTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_data,
        peft_config=peft_config,
        eval_dataset=test_data,
        weight=args.weight,
        eval_batch_size=args.per_device_eval_batch_size,
    )
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(torch.cuda.memory_summary())
    trainer.train()
    if args.report_to == "wandb":
        wandb.finish()
    trainer.save_model(unique_output_dir)
    print(f"Model saved to {unique_output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/mnt/shared-research-data/reward_model_group")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=12)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--lora_rank", type=int, default=8)
    parser.add_argument("--num_rows", type=int, default=30000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="wandb")
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)

