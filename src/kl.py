import argparse
import os
import json
import random
import torch
import wandb
import gzip
import shutil
import requests
import csv
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from trl import DPOConfig, DPOTrainer
from peft import LoraConfig
import pandas as pd
from load_huggingface import load_huggingface
from typing import Union, Literal

os.environ["TOKENIZERS_PARALLELISM"] = "false"
random.seed(42)
torch.manual_seed(42)
torch.cuda.empty_cache()

def prompt(text):
    last_assistant = text.rfind("Assistant:")
    return text[:last_assistant].strip()

def answer(text):
    last_assistant = text.rfind("Assistant:")
    return text[last_assistant:].strip()

def process_default(dataset):
    df = pd.DataFrame({
        "prompt": [prompt(text) for text in dataset['chosen']],
        "chosen": [answer(text) for text in dataset['chosen']],
        "rejected": [answer(text) for text in dataset['rejected']],
    })
    df["row_id"] = df.index.values
    return Dataset.from_pandas(df)

def tokenize_fn(batch, tokenizer, max_length):
    prompt = tokenizer(
        batch["prompt"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )

    chosen = tokenizer(
        batch["chosen"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    rejected = tokenizer(
        batch["rejected"],
        truncation=True,
        padding="max_length",
        max_length=max_length,
        return_attention_mask=True,
    )
    return {
        "prompt_input_ids": prompt["input_ids"],
        "prompt_attention_mask": prompt["attention_mask"],
        "chosen_input_ids": chosen["input_ids"],
        "chosen_attention_mask": chosen["attention_mask"],
        "rejected_input_ids": rejected["input_ids"],
        "rejected_attention_mask": rejected["attention_mask"],
    }

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


class PerExampleLogProbsCallback(TrainerCallback):
    def __init__(self, output_dir, tokenizer):
        self.output_dir = output_dir
        self.tokenizer = tokenizer
        os.makedirs(self.output_dir, exist_ok=True)

    def set_trainer(self, trainer):
        self.trainer = trainer

    def on_train_begin(self, args, state, control, **kwargs):
        if not hasattr(self, "trainer") or self.trainer is None:
            print("Trainer reference not set; skipping initial logging.")
            return control

        train_dataset = self.trainer.train_dataset
        initial_data = []

        for i in range(len(train_dataset)):
            initial_data.append({
                "row_id": i, 
                "logps_chosen_epoch_0": train_dataset[i]["ref_chosen_logps"],
                "logps_rejected_epoch_0": train_dataset[i]["ref_rejected_logps"],
            })
        initial_df = pd.DataFrame(initial_data)

        reference_filename = os.path.join(self.output_dir, "epoch_0.csv")
        initial_df.to_csv(reference_filename, index=False)
        print(f"Saved reference log probabilities to {reference_filename}")

    def on_epoch_end(self, args, state, control, **kwargs):
        if not hasattr(self, "trainer") or self.trainer is None:
            print("Trainer reference not set; skipping logging.")
            return control

        model = self.trainer.model
        model.eval()
        dataloader = self.trainer.get_train_dataloader()
        epoch_data = []
        example_id = 0

        for batch in dataloader:
            for key, value in batch.items():
                if isinstance(value, torch.Tensor):
                    batch[key] = value.to(next(model.parameters()).device)

            with torch.no_grad():
                model_output = self.trainer.concatenated_forward(model, batch)
                chosen_logps = model_output["chosen_logps"].detach().cpu()
                rejected_logps = model_output["rejected_logps"].detach().cpu()

                batch_size = chosen_logps.size(0)
                for i in range(batch_size):
                    epoch_data.append({
                        "row_id": example_id,
                        f"logps_chosen_epoch_{state.epoch}": chosen_logps[i].item(),
                        f"logps_rejected_epoch_{state.epoch}": rejected_logps[i].item(),
                    })
                    example_id += 1

        epoch_df = pd.DataFrame(epoch_data)
        device = next(model.parameters()).device
        device_name = torch.cuda.get_device_name(device.index) if torch.cuda.is_available() else "cpu"
        epoch_filename = os.path.join(
            self.output_dir,
            f"epoch_{state.epoch}_{device_name}.csv"
        )

        epoch_df.to_csv(epoch_filename, index=False)
        print(f"Saved per-example log probabilities for epoch {state.epoch} on {device_name} to {epoch_filename}")

        model.train()
        return control

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")
    if args.report_to == "wandb":
        wandb.init(project="dpo_training", config=args.__dict__)

    base_url = "https://huggingface.co/datasets/Anthropic/hh-rlhf/resolve/main/helpful-base/"
    for file in ["train.jsonl.gz", "test.jsonl.gz"]:
        download_and_decompress(base_url + file, file.replace(".gz", ""))
    train_data = Dataset.from_list(load_jsonl("train.jsonl"))
    test_data = Dataset.from_list(load_jsonl("test.jsonl"))
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    if args.num_rows > 0:
        train_data = train_data.select(range(min(len(train_data), args.num_rows)))
    if args.test_size > 0:
        test_data = test_data.select(range(min(len(test_data), args.test_size)))

    train_data = process_default(train_data)
    test_data = process_default(test_data)

    train_data = train_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)
    test_data = test_data.map(lambda batch: tokenize_fn(batch, tokenizer, args.max_length), batched=True)

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
        eval_dataset=test_data,
        peft_config=peft_config,
    )
    per_example_logger = PerExampleLogProbsCallback(output_dir=args.output_dir, tokenizer=tokenizer)
    trainer.add_callback(per_example_logger)
    per_example_logger.set_trainer(trainer)

    trainer.train()

    if args.report_to == "wandb":
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="gpt2")
    parser.add_argument("--output_dir", type=str, default="./dpo_model")
    parser.add_argument("--per_device_train_batch_size", type=int, default=10)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--max_length", type=int, default=1500)
    parser.add_argument("--beta", type=float, default=0.01)
    parser.add_argument("--lora_rank", type=int, default=12)
    parser.add_argument("--num_rows", type=int, default=20)
    parser.add_argument("--test_size", type=int, default=10)
    parser.add_argument("--report_to", type=str, choices=["none", "wandb"], default="none")
    parser.add_argument("--logging_steps", type=int, default=100)
    args = parser.parse_args()
    main(args)