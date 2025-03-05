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
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, TrainerCallback
from trl import PPOConfig, PPOTrainer
from peft import LoraConfig

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.manual_seed(42)
random.seed(42)
torch.cuda.empty_cache()

def download_and_decompress(url, output_file):
    compressed_file = output_file + ".gz"
    response = requests.get(url)
    if response.status_code == 200:
        with open(compressed_file, "wb") as f:
            f.write(response.content)
        with gzip.open(compressed_file, "rb") as f_in, open(output_file, "wb") as f_out:
            shutil.copyfileobj(f_in, f_out)
    else:
        raise Exception(f"Failed to download {compressed_file}: HTTP {response.status_code}")

def load_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def extract_prompt(text):
    last_assistant = text.rfind("Assistant:")
    return text[:last_assistant].strip() if last_assistant != -1 else text.strip()

def process_data(dataset):
    data_list = dataset.to_list()
    df = pd.DataFrame({
        "query": [extract_prompt(entry["chosen"]) for entry in data_list],
    })
    return Dataset.from_pandas(df)

def tokenize_fn(batch, tokenizer, max_length):
    prompt = tokenizer(batch["query"], truncation=True, padding="max_length", max_length=max_length, return_attention_mask=True)

    return {
        "input_ids": prompt["input_ids"],
        "attention_mask": prompt["attention_mask"],
    }

def sample_and_shuffle(data_helpful, data_harmless, num_samples, weight):
    if num_samples > 0:
        n_helpful = min(int(weight * num_samples), len(data_helpful))
        n_harmless = min(num_samples - n_helpful, len(data_harmless))
        sampled_helpful = data_helpful.select(random.sample(range(len(data_helpful)), n_helpful))
        sampled_harmless = data_harmless.select(random.sample(range(len(data_harmless)), n_harmless))
        combined_data = [{"dID": 0, **entry} for entry in sampled_helpful.to_list()] + \
                        [{"dID": 1, **entry} for entry in sampled_harmless.to_list()]
        random.shuffle(combined_data)
        return Dataset.from_list(combined_data)
    return Dataset.from_list([])

class CustomEvalCallback(TrainerCallback):
    def __init__(self, ppo_trainer, eval_dataset_helpful, eval_dataset_harmless):
        self.ppo_trainer = ppo_trainer
        self.eval_dataset_helpful = eval_dataset_helpful
        self.eval_dataset_harmless = eval_dataset_harmless
    
    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % args.logging_steps == 0:
            print(f"Evaluating at step {state.global_step}...")
            helpful_rewards = self.evaluate_dataset(self.eval_dataset_helpful, "helpful")
            harmless_rewards = self.evaluate_dataset(self.eval_dataset_harmless, "harmless")
            if args.report_to == "wandb":
                wandb.log({
                    "eval/helpful_reward_mean": sum(helpful_rewards) / len(helpful_rewards) if helpful_rewards else 0,
                    "eval/harmless_reward_mean": sum(harmless_rewards) / len(harmless_rewards) if harmless_rewards else 0,
                    "step": state.global_step,
                })
    
    def evaluate_dataset(self, dataset, name):
        rewards = []
        sample_size = min(50, len(dataset))
        for idx in random.sample(range(len(dataset)), sample_size):
            entry = dataset[idx]
            query_tensor = self.ppo_trainer.tokenizer(entry["query"], return_tensors="pt", padding=True, truncation=True).to(self.ppo_trainer.accelerator.device)
            response_tensor = self.ppo_trainer.generate(
                query_tensor.input_ids,
                max_new_tokens=self.ppo_trainer.config.response_length,
                do_sample=True,
                temperature=self.ppo_trainer.config.temperature
            )
            response_text = self.ppo_trainer.tokenizer.decode(
                response_tensor[0][query_tensor.input_ids.shape[1]:],
                skip_special_tokens=True
            )
            with torch.no_grad():
                reward_input = self.ppo_trainer.tokenizer(
                    [entry["query"] + response_text],
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.ppo_trainer.accelerator.device)
                reward_score = self.ppo_trainer.reward_model(**reward_input).logits.item()
                rewards.append(reward_score)
        print(f"{name.capitalize()} mean reward: {sum(rewards) / len(rewards):.4f}" if rewards else f"{name.capitalize()} dataset empty")
        return rewards

def get_rewards_from_model(queries, responses, reward_model, tokenizer, device, max_length=512):
    # Formula: reward = reward_model(tokenizer(query + response)).logits
    inputs = [q + r for q, r in zip(queries, responses)]
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = reward_model(**tokenized)
        rewards = outputs.logits.squeeze(-1)
    return rewards

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.report_to == "wandb":
        wandb.init(project="ppo_training", config=vars(args))
    
    # Download and load datasets (using helpful and harmless splits)
    base_helpful = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/"
    for file in ["train.jsonl.gz", "test.jsonl.gz"]:
        download_and_decompress(base_helpful + file, file.replace(".gz", ""))
    train_helpful = Dataset.from_list(load_jsonl("train.jsonl"))
    test_helpful = Dataset.from_list(load_jsonl("test.jsonl"))
    
    base_harmless = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/harmless-base/"
    for file in ["train.jsonl.gz", "test.jsonl.gz"]:
        download_and_decompress(base_harmless + file, file.replace(".gz", ""))
    train_harmless = Dataset.from_list(load_jsonl("train.jsonl"))
    test_harmless = Dataset.from_list(load_jsonl("test.jsonl"))
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_data = sample_and_shuffle(train_helpful, train_harmless, args.num_rows, args.weight)
    test_data_helpful = sample_and_shuffle(test_helpful, Dataset.from_list([]), args.test_size, 1)
    test_data_harmless = sample_and_shuffle(Dataset.from_list([]), test_harmless, args.test_size, 0)
    train_data = process_data(train_data)
    test_data_helpful = process_data(test_data_helpful)
    test_data_harmless = process_data(test_data_harmless)

    train_data = train_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True, remove_columns=["query"])
    test_data_harmless = test_data_harmless.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True, remove_columns=["query"])
    
    reward_model = AutoModelForSequenceClassification.from_pretrained(args.reward_model_path).to(device)
    reward_model.eval()
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    
    ppo_config = PPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_ppo_epochs=args.ppo_epochs,
        response_length=args.response_length,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
    )
    
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=model,           
        ref_model=ref_model,  
        value_model=model,    
        reward_model=reward_model, 
        processing_class=tokenizer,
        train_dataset=train_data,
    )

    
    ppo_trainer.add_callback(CustomEvalCallback(ppo_trainer, test_data_helpful, test_data_harmless))    
    ppo_trainer.train()
    ppo_trainer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")
    if args.report_to == "wandb":
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--reward_model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./ppo_model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--mini_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--response_length", type=int, default=128)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--use_lora", action="store_true")
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="none")
    # Data sampling parameters (essential for dataset preparation)
    parser.add_argument("--num_rows", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--weight", type=float, default=0.8)
    parser.add_argument("--logging_steps", type=int, default=10)
    args = parser.parse_args()
    main(args)
