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
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
from datasets import concatenate_datasets

os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)
torch.manual_seed(42)
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

def prompt(text):
    last_assistant = text.rfind("Assistant:")
    return text[:last_assistant].strip()

def answer(text):
    last_assistant = text.rfind("Assistant:")
    return text[last_assistant:].strip()

def process_default(dataset):
    data_list = dataset.to_list()
    df = pd.DataFrame({
        "prompt": [prompt(text["chosen"]) for text in data_list],
        "chosen": [answer(text["chosen"]) for text in data_list],
        "rejected": [answer(text["rejected"]) for text in data_list],
        "dID": [text["dID"] for text in data_list],
    })
    df["row_id"] = df.index.values
    return Dataset.from_pandas(df)

def tokenize_fn(batch, tokenizer, max_length):
    formatted_prompts = [tokenizer.apply_chat_template([{"role": "user", "content": p}], tokenize=False) for p in batch["prompt"]]
    formatted_prompts = [p for p in batch["prompt"]]
    formatted_chosen = [
        tokenizer.apply_chat_template([
            {"role": "user", "content": batch["prompt"][i]}, 
            {"role": "assistant", "content": batch["chosen"][i]}
        ], tokenize=False) 
        for i in range(len(batch["prompt"]))
    ]
    formatted_rejected = [
        tokenizer.apply_chat_template([
            {"role": "user", "content": batch["prompt"][i]}, 
            {"role": "assistant", "content": batch["rejected"][i]}
        ], tokenize=False) 
        for i in range(len(batch["prompt"]))
    ]
    
    prompt = tokenizer(formatted_prompts, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    chosen = tokenizer(formatted_chosen, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    rejected = tokenizer(formatted_rejected, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")

    return {
        "row_id": batch["row_id"],
        "dID": batch["dID"],
        "prompt_input_ids": prompt["input_ids"],
        "prompt_attention_mask": prompt["attention_mask"],
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
    }

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

class CustomEvalCallback(TrainerCallback):
    def __init__(self, trainer, eval_dataset_helpful, eval_dataset_harmless):
        self.trainer = trainer
        self.eval_dataset_helpful = eval_dataset_helpful
        self.eval_dataset_harmless = eval_dataset_harmless
    
    def on_evaluate(self, args, state, control, **kwargs):
        results_helpful = self.trainer.evaluate(eval_dataset=self.eval_dataset_helpful)
        results_harmless = self.trainer.evaluate(eval_dataset=self.eval_dataset_harmless)
        
        if args.report_to == "wandb":
            wandb.log({"helpfulness_eval_loss": results_helpful["eval_loss"],
                       "harmlessness_eval_loss": results_harmless["eval_loss"]})
            
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    if args.report_to == "wandb":
        wandb.init(project="dpo_group_training", config=args.__dict__)

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
    tokenizer.pad_token = tokenizer.eos_token

    train_data = sample_and_shuffle(train_data_helpful, train_data_harmless, args.num_rows, args.weight)
    test_data_helpful = sample_and_shuffle(test_data_helpful, Dataset.from_list([]), args.test_size, 1)
    test_data_harmless = sample_and_shuffle(Dataset.from_list([]), test_data_harmless, args.test_size, 0)

    train_data = process_default(train_data)
    test_data_helpful = process_default(test_data_helpful)
    test_data_harmless = process_default(test_data_harmless)

    train_data = train_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)
    test_data_helpful = test_data_helpful.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)
    test_data_harmless = test_data_harmless.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)
    test_data = concatenate_datasets([test_data_helpful, test_data_harmless])

    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=torch.float16).to(device)
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

    training_args = DPOConfig(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.per_device_train_batch_size,
        max_length=args.max_length,
        precompute_ref_log_probs=True,
        remove_unused_columns=False,
        beta=args.beta,
        logging_steps=args.logging_steps,
        num_train_epochs=args.epochs,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        eval_strategy="steps",
        eval_steps=100,
        fp16=torch.cuda.is_available(),
    )

    trainer = DPOTrainer(
        model=model,
        # ref_model=ref_model,
        args=training_args,
        tokenizer=tokenizer,
        train_dataset=train_data,
        peft_config=peft_config,
        eval_dataset=test_data,
    )

    eval_callback = CustomEvalCallback(trainer, test_data_helpful, test_data_harmless)
    trainer.add_callback(eval_callback)

    trainer.train()

    if args.report_to == "wandb":
        wandb.finish()

    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./dpo_model_group")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--beta", type=float, default=0.0001)
    parser.add_argument("--lora_rank", type=int, default=12)
    parser.add_argument("--num_rows", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="wandb")
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--weight", type=float, default=0.8)
    args = parser.parse_args()

    main(args)
