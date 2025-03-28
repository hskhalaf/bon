import argparse
import os
from unsloth import FastLanguageModel 
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
import numpy as np
from accelerate import Accelerator
import torch.nn.functional as F

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
    formatted_chosen = [
        tokenizer.apply_chat_template([
            {"role": "assistant", "content": batch["chosen"][i]}
        ], tokenize=False) 
        for i in range(len(batch["chosen"]))
    ]
    formatted_rejected = [
        tokenizer.apply_chat_template([
            {"role": "assistant", "content": batch["rejected"][i]}
        ], tokenize=False) 
        for i in range(len(batch["rejected"]))
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

def compute_log_probs_on_answer(model, prompt_ids, prompt_mask, answer_ids, answer_mask):
    input_ids = torch.cat([prompt_ids, answer_ids], dim=1)
    attention_mask = torch.cat([prompt_mask, answer_mask], dim=1)

    with torch.no_grad():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    shift_mask = attention_mask[:, 1:]

    log_probs = F.log_softmax(shift_logits, dim=-1)
    selected_log_probs = torch.gather(log_probs, dim=2, index=shift_labels.unsqueeze(-1)).squeeze(-1)

    prompt_lengths = prompt_mask.sum(dim=1)
    full_mask = torch.zeros_like(shift_mask)
    for i, start in enumerate(prompt_lengths):
        full_mask[i, start-1:] = 1

    answer_log_probs = selected_log_probs * full_mask
    return answer_log_probs.sum(dim=1)


def compute_custom_metric(eval_dataset, model, batch_size):
    device = model.device
    model.eval()
    total_examples = len(eval_dataset)
    all_chosen_scores = []
    all_rejected_scores = []
    for i in range(0, total_examples, batch_size):
        batch_indices = range(i, min(i + batch_size, total_examples))
        prompt_ids = torch.stack([torch.tensor(eval_dataset[j]["prompt_input_ids"]) for j in batch_indices]).to(device)
        prompt_mask = torch.stack([torch.tensor(eval_dataset[j]["prompt_attention_mask"]) for j in batch_indices]).to(device)
        chosen_ids = torch.stack([torch.tensor(eval_dataset[j]["chosen_input_ids"]) for j in batch_indices]).to(device)
        chosen_mask = torch.stack([torch.tensor(eval_dataset[j]["chosen_attention_mask"]) for j in batch_indices]).to(device)
        rejected_ids = torch.stack([torch.tensor(eval_dataset[j]["rejected_input_ids"]) for j in batch_indices]).to(device)
        rejected_mask = torch.stack([torch.tensor(eval_dataset[j]["rejected_attention_mask"]) for j in batch_indices]).to(device)

        rewards_chosen = compute_log_probs_on_answer(model, prompt_ids, prompt_mask, chosen_ids, chosen_mask)
        rewards_rejected = compute_log_probs_on_answer(model, prompt_ids, prompt_mask, rejected_ids, rejected_mask)

        all_chosen_scores.append(rewards_chosen.cpu())
        all_rejected_scores.append(rewards_rejected.cpu())
    all_chosen_scores = torch.cat(all_chosen_scores).tolist()
    all_rejected_scores = torch.cat(all_rejected_scores).tolist()
    results = [all_chosen_scores[i] > all_rejected_scores[i] for i in range(len(all_chosen_scores))]
    torch.cuda.empty_cache()
    return sum(results) / len(results) if results else 0

class CustomDPOTrainer(DPOTrainer):
    def __init__(self, *args, eval_dataset_helpful=None, eval_dataset_harmless=None, weight=1.0, eval_batch_size=8, **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_dataset_helpful = eval_dataset_helpful
        self.eval_dataset_harmless = eval_dataset_harmless
        self.weight = weight
        self.eval_batch_size = eval_batch_size

    def evaluate(self, eval_dataset=None, **kwargs):
        helpful = compute_custom_metric(self.eval_dataset_helpful, self.model, self.eval_batch_size)
        harmless = compute_custom_metric(self.eval_dataset_harmless, self.model, self.eval_batch_size)
        metrics = {"eval_helpful_accuracy": helpful, "eval_harmless_accuracy": harmless, "eval_avg": helpful * self.weight + harmless * (1 - self.weight)}
        self.log(metrics)
        return metrics
     
def main(args):
    print("using", torch.cuda.device_count(), "GPUs!")
    base_output_dir = args.output_dir
    output_dir = os.path.join(base_output_dir, f"{args.model_name.replace('/', '_')}_seed{args.seed}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving outputs to: {output_dir}")
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

    model, tokenizer = FastLanguageModel.from_pretrained(
            model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
            max_seq_length = args.max_length,
            dtype=torch.bfloat16 if use_bf16 else torch.float32,
            device_map = "auto",
    )

    # ref_model = FastLanguageModel.from_pretrained(
    #         model_name = "unsloth/Meta-Llama-3.1-8B-Instruct",
    #         max_seq_length = args.max_length,
    #         torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    #         device_map = "auto",
    # )

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
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

    # accelerator = Accelerator()
    # model = AutoModelForCausalLM.from_pretrained(
    #     args.model_name, 
    #     torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    #     device_map={"": accelerator.local_process_index},)
    
    # ref_model = AutoModelForCausalLM.from_pretrained(
    #      args.model_name, 
    #     torch_dtype=torch.bfloat16 if use_bf16 else torch.float32,
    #     device_map={"": accelerator.local_process_index},
    # )


    model = FastLanguageModel.get_peft_model(
        model,
        r = args.lora_rank,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj",],
        lora_alpha = 16,
        lora_dropout = 0.1, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
        random_state = args.seed,
        max_seq_length = args.max_length,
    )

    
    # for param in ref_model.parameters():
    #     param.requires_grad = False

    # peft_config = LoraConfig(
    #     task_type="CAUSAL_LM",
    #     inference_mode=False,
    #     r=args.lora_rank,
    #     lora_alpha=16,
    #     lora_dropout=0.1,
    # )

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
        eval_steps=args.logging_steps,
        fp16=use_fp16,
        bf16=use_bf16,
        seed=args.seed,
        data_seed=args.seed,
        save_steps=300,
        save_total_limit=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_avg",
        greater_is_better=True,
    )

    dummy_test = test_data_harmless.select([0])

    trainer = CustomDPOTrainer(
        model=model,
        args=training_args,
        processing_class=tokenizer,
        train_dataset=train_data,
        # peft_config=peft_config,
        eval_dataset=dummy_test,
        eval_dataset_helpful=test_data_helpful,
        eval_dataset_harmless=test_data_harmless,
    )

    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    print(torch.cuda.memory_summary())
    trainer.train()

    if args.report_to == "wandb":
        wandb.finish()

    trainer.save_model(args.output_dir)
    print(f"Model saved to {args.output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct")
    parser.add_argument("--output_dir", type=str, default="/mnt/shared-research-data/dpo/")
    parser.add_argument("--per_device_train_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--beta", type=float, default=0.1)
    parser.add_argument("--lora_rank", type=int, default=12)
    parser.add_argument("--num_rows", type=int, default=30000)
    parser.add_argument("--test_size", type=int, default=1000)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="wandb")
    parser.add_argument("--logging_steps", type=int, default=30)
    parser.add_argument("--weight", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    main(args)
