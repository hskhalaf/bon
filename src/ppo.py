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
from peft import LoraConfig, PeftModel, get_peft_model
import numpy as np
from accelerate import Accelerator
os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

def process_data(dataset):
    data_list = dataset.to_list()
    df = pd.DataFrame({
        "query": [process_conversation(extract_prompt(entry["chosen"]) + " Assistant:") for entry in data_list],
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
            torch.cuda.empty_cache()
            if args.report_to == "wandb":
                wandb.log({
                    "eval/helpful_reward_mean": sum(helpful_rewards) / len(helpful_rewards) if helpful_rewards else 0,
                    "eval/harmless_reward_mean": sum(harmless_rewards) / len(harmless_rewards) if harmless_rewards else 0,
                    "step": state.global_step,
                })
    
    def evaluate_dataset(self, dataset, name):
        rewards = []
        sample_size = min(50, len(dataset))
        batch_size = 8
        for start_idx in range(0, sample_size, batch_size):
            end_idx = min(start_idx + batch_size, sample_size)
            batch_indices = random.sample(range(len(dataset)), end_idx - start_idx)
            batch_entries = [dataset[idx] for idx in batch_indices]
            
            batch_queries = [entry["query"] for entry in batch_entries]
            batch_query_tensors = self.ppo_trainer.tokenizer(batch_queries, return_tensors="pt", padding=True, truncation=True).to(self.ppo_trainer.accelerator.device)
            
            batch_response_tensors = []
            for query_tensor in batch_query_tensors.input_ids:
                response_tensor = self.ppo_trainer.generate(
                    query_tensor.unsqueeze(0),
                    max_new_tokens=self.ppo_trainer.config.response_length,
                    do_sample=True,
                    temperature=self.ppo_trainer.config.temperature
                )
                batch_response_tensors.append(response_tensor[0])
            
            batch_responses = []
            for i, response_tensor in enumerate(batch_response_tensors):
                query_length = batch_query_tensors.input_ids[i].shape[0]
                response_text = self.ppo_trainer.tokenizer.decode(
                    response_tensor[query_length:],
                    skip_special_tokens=True
                )
                batch_responses.append(response_text)
            
            batch_inputs = [q + r for q, r in zip(batch_queries, batch_responses)]
            with torch.no_grad():
                reward_inputs = self.ppo_trainer.tokenizer(
                    batch_inputs,
                    return_tensors="pt",
                    padding=True,
                    truncation=True
                ).to(self.ppo_trainer.accelerator.device)
                reward_scores = self.ppo_trainer.reward_model(**reward_inputs).logits.squeeze(-1).tolist()
                rewards.extend(reward_scores)
        
        print(f"{name.capitalize()} mean reward: {sum(rewards) / len(rewards):.4f}" if rewards else f"{name.capitalize()} dataset empty")
        return rewards

def get_rewards_from_model(queries, responses, reward_model, tokenizer, device, max_length=512):
    inputs = [q + r for q, r in zip(queries, responses)]
    tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
    with torch.no_grad():
        outputs = reward_model(**tokenized)
        rewards = outputs.logits.squeeze(-1)
    return rewards

def collator(data):
    return {key: [d[key] for d in data] for key in data[0]}

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
        wandb.init(project="ppo_training", config=args.__dict__)
    
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
    
    adapter_path = args.reward_model_path
    config_path = os.path.join(adapter_path, "adapter_config.json")
    with open(config_path, 'r') as f:
        adapter_config = json.load(f)
    reward_model_name = adapter_config["base_model_name_or_path"]

    reward_tokenizer = AutoTokenizer.from_pretrained(reward_model_name)
    if reward_tokenizer.pad_token is None:
        reward_tokenizer.pad_token = reward_tokenizer.eos_token
    
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
    test_data_helpful = test_data_helpful.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True, remove_columns=["query"])
    test_data_harmless = test_data_harmless.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True, remove_columns=["query"])
    
    accelerator = Accelerator()
    base_reward_model = AutoModelForSequenceClassification.from_pretrained(
        reward_model_name, 
        num_labels = 1,  
        torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
        device_map={"": accelerator.local_process_index},
    )
    
    reward_model = PeftModel.from_pretrained(base_reward_model, adapter_path, device_map={"": accelerator.local_process_index},)
    reward_model.eval()
    reward_model.config.pad_token_id = reward_tokenizer.pad_token_id

    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_ref_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_ref_model.config.pad_token_id = tokenizer.pad_token_id
    peft_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, peft_config)
    ref_model = base_ref_model
    
    ppo_config = PPOConfig(
        output_dir=args.output_dir,
        learning_rate=args.learning_rate,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_ppo_epochs=args.ppo_epochs,
        response_length=args.response_length,
        temperature=args.temperature,
        kl_coef=args.kl_coef,
        max_grad_norm=1.0,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        save_steps=300,
        save_total_limit=5,
        seed = args.seed,
        data_seed = args.seed,
        report_to=args.report_to,
        fp16=use_fp16,
        bf16=use_bf16,
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
    
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(torch.cuda.memory_summary())
    ppo_trainer.add_callback(CustomEvalCallback(ppo_trainer, test_data_helpful, test_data_harmless))    
    ppo_trainer.train()
    
    ppo_trainer.save_model(unique_output_dir)
    print(f"Model saved to {unique_output_dir}")
    
    if args.report_to == "wandb":
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--reward_model_path", type=str, default="Qwen/Qwen2.5-0.5B")
    parser.add_argument("--output_dir", type=str, default="./ppo_model")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--ppo_epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--response_length", type=int, default=128)
    parser.add_argument("--kl_coef", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--lora_rank", type=int, default=16)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="none")
    parser.add_argument("--num_rows", type=int, default=1000)
    parser.add_argument("--test_size", type=int, default=100)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--logging_steps", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20)
    args = parser.parse_args()
    main(args)