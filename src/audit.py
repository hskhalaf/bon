import os
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
import pandas as pd
from tqdm import tqdm
import argparse
import re 

from load_huggingface import load_huggingface
from prompt_template import form_pairwise_chat_prompt

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DEFAULTS = {
    "dataset_name": "Anthropic/hh-rlhf",
    "attributes_name": "prompts.txt",
    "model_name": "llama8b-i",
    "start_row": None,
    "end_row": 5,
    "split": 'test',
    "batch_size": 5,
    "max_length": 2500,
    "torch_dtype": torch.float16,
    "devices": 'mps',
    "output": 'values.csv'
}
MODEL_INFO_MAP = {
    'qwen1b-i': {
        'model_id': 'Qwen/Qwen2.5-1.5B-Instruct',
    },
    'llama8b-i': {
        'model_id': "meta-llama/Llama-3.1-8B-Instruct",
    },
    'llama8b': {
        'model_id': "meta-llama/Llama-3.1-8B",
    },
    'llama70b': {
        'model_id': "meta-llama/Llama-3.1-70B",
    },
    'llama70b-i': {
        'model_id': "meta-llama/Llama-3.1-70B-Instruct",
    },
}


def main(config):
    mps_available = torch.backends.mps.is_available()
    if torch.cuda.is_available() and config['devices'] is None:
        devices = ["cuda:0", "cuda:1"] if torch.cuda.device_count() > 1 else "cuda:0"
    elif mps_available and config['devices'] is None:
        devices = 'mps'
    elif config['devices'] is None:
        devices = "cpu"
    else:
        devices = config['devices']  # Use the provided devices
    config['devices'] = devices
    config['devices'] = devices

    if config['model_name'] in MODEL_INFO_MAP:
        config |= MODEL_INFO_MAP[config['model_name']]
    else:
        config['model_id'] = config['model_name']

    assert config['output'].endswith('.csv'), "Output file must have a .csv extension"
    if not config['output'].startswith('/'):
        output_dir = "output"
        config['output'] = os.path.join(ROOT_DIR, output_dir, config['output'])
        os.makedirs(os.path.dirname(config['output']), exist_ok=True)

    if config['attributes_name'] is not None:
        if not os.path.exists(config['attributes_name']):
            config['attributes_name'] = os.path.join(ROOT_DIR, "prompts", config['attributes_name'])
        with open(config['attributes_name'], 'r') as file:
            attributes = file.read().splitlines()
            attributes = dict(enumerate(attributes))

    df = load_huggingface(**config)

    if config['start_row'] is not None:
        df = df[df['row_id'] >= config['start_row']]
    if config['end_row'] is not None:
        df = df[df['row_id'] < config['end_row']]
    prompts_df = compose_prompts(df, attributes)
    
    if 'model_id' not in config:
        print("Error: 'model_id' is not set in the configuration.")
        return

    try:
        tokenizer = AutoTokenizer.from_pretrained(config['model_id'])
    except Exception as e:
        print(f"Error loading tokenizer for model {config['model_id']}: {e}")
        return
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenized_prompts = tokenize_prompts(prompts_df, tokenizer, config)
    tokenized_dataset = TokenizedPromptDataset(tokenized_prompts)
    result_df = annotate_prompts(tokenized_dataset, tokenizer, config)

    if not result_df.empty :
        pivot_df = result_df.pivot_table(
            index='row_id',
            columns='attribute_id',
            values=['prob_A', 'prob_B', 'prob_NA']
        )

        pivot_df.columns = [f'{value}_p{attribute}' for value, attribute in pivot_df.columns]
        pivot_df = pivot_df.reset_index()

        result_df = pivot_df.merge(df, on='row_id').drop(
            columns=[col for col in pivot_df.merge(df, on='row_id').columns if col.endswith('_dupe')],
            errors='ignore'
        )
        
        result_df.to_csv(config['output'].replace('.csv', '_processed.csv'), index=False)
    else:
        print("No results to process.")

def compose_prompts(df, attributes):
    def compose_prompts_for_row(row):
        batch_prompts = []
        row_ids = []
        attribute_ids = []
        attributes_to_use = attributes

        for attribute_id, attribute in attributes_to_use.items():
            x = row['prompt']
            y_a = row['chosen']
            y_b = row['rejected']

            
            prompt = form_pairwise_chat_prompt(x, y_a, y_b, attribute)
            batch_prompts.append(prompt)
            row_ids.append(row['row_id'])
            attribute_ids.append(attribute_id)


        return pd.DataFrame({
            'prompt': batch_prompts,
            'row_id': row_ids,
            'attribute_id': attribute_ids,
        })

    prompts_list = [compose_prompts_for_row(row) for _, row in df.iterrows()]
    prompts_df = pd.concat(prompts_list, ignore_index=True)
    return prompts_df


def tokenize_prompts(prompts_df, tokenizer, config):
    tokenized_data = {"input_ids": [], "attention_mask": []}

    max_prompt_length = config['max_length']
    for i in range(0, len(prompts_df), config['batch_size']):
        batch_df = prompts_df.iloc[i:i + config['batch_size']]
        batch_prompts = []
        for prompt in batch_df['prompt']:
            prompt = tokenizer.apply_chat_template(prompt, tokenize=False, add_generation_prompt=True)
            batch_prompts.append(prompt)
        
        prompt_inputs = tokenizer(batch_prompts, return_tensors='pt', padding='max_length', truncation=True, max_length=max_prompt_length)

        if prompt_inputs['attention_mask'].size(0) > 0 and prompt_inputs['attention_mask'][:, -1].any().item():
            raise ValueError("Last token in input_ids is not masked, which may suggest max_length is too low!")
        
        tokenized_data['input_ids'].extend(prompt_inputs['input_ids'])
        tokenized_data['attention_mask'].extend(prompt_inputs['attention_mask'])

    tokenized_data["input_ids"] = torch.stack(tokenized_data["input_ids"])
    tokenized_data["attention_mask"] = torch.stack(tokenized_data["attention_mask"])
    tokenized_data["row_ids"] = torch.tensor(prompts_df['row_id'].values)
    tokenized_data["attribute_ids"] = torch.tensor(prompts_df['attribute_id'].values)
    return tokenized_data


class TokenizedPromptDataset(Dataset):
    def __init__(self, tokenized_data):
        self.input_ids = tokenized_data["input_ids"]
        self.attention_mask = tokenized_data["attention_mask"]
        self.row_ids = tokenized_data["row_ids"]
        self.attribute_ids = tokenized_data["attribute_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "row_id": self.row_ids[idx],
            "attribute_id": self.attribute_ids[idx],
        }


def annotate_prompts(dataset, tokenizer, config, existing_result_df=None):
    devices = config['devices']

    torch_dtype = config['torch_dtype']
    if (len(devices) == 1) and devices[0] == 'cpu':
        torch_dtype = torch.float32

    target_token_A = "A"
    target_token_B = "B"
    target_token_NA = "NA"
    target_token_id_A = tokenizer.encode(target_token_A, add_special_tokens=False)[0]
    target_token_id_B = tokenizer.encode(target_token_B, add_special_tokens=False)[0]
    target_token_id_NA = tokenizer.encode(target_token_NA, add_special_tokens=False)[0]

    try:
        model = AutoModelForCausalLM.from_pretrained(config['model_id'], torch_dtype=torch_dtype, device_map=devices)
    except Exception as e:
        print(f"Error loading model for model {config['model_id']}: {e}")
        return
    model.eval()
    loader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    full_score_df = existing_result_df
    all_scores = []
    for batch in tqdm(loader, desc=f"Processing Batches on {devices}"):
        row_ids = batch["row_id"]
        attribute_ids = batch["attribute_id"]
        first_device = next(model.parameters()).device
        inputs = {key: val.to(first_device) for key, val in batch.items() if key in ["input_ids", "attention_mask"]}

        with torch.no_grad():
            outputs = model(**inputs) 
            logits = outputs.logits

        last_non_padding_idx = inputs["attention_mask"].sum(dim=1).sub(1)
        next_token_logits = logits[torch.arange(logits.size(0)), last_non_padding_idx, :]

        probs = torch.softmax(next_token_logits, dim=-1)
        scores_batch_A = probs[:, target_token_id_A].cpu()
        scores_batch_B = probs[:, target_token_id_B].cpu()
        scores_batch_NA = probs[:, target_token_id_NA].cpu()
        scores_df = pd.DataFrame({
                    'row_id': row_ids.numpy(),
                    'attribute_id': attribute_ids.numpy(),
                    'prob_A': scores_batch_A.numpy(),
                    'prob_B': scores_batch_B.numpy(),
                    'prob_NA': scores_batch_NA.numpy()
                })
        all_scores.append(scores_df)
    full_score_df = pd.concat(all_scores, ignore_index=True) if all_scores else pd.DataFrame()
    if not full_score_df.empty: 
        full_score_df.to_csv(config['output'], index=False)
    return full_score_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--attributes_name", nargs='?', default=DEFAULTS["attributes_name"])
    parser.add_argument("--dataset_name", nargs='?', default=DEFAULTS["dataset_name"])
    parser.add_argument("--model_name", nargs='?', default=DEFAULTS["model_name"])
    parser.add_argument("--start_row", nargs='?', type=int, default=DEFAULTS["start_row"])
    parser.add_argument("--end_row", nargs='?', type=int, default=DEFAULTS["end_row"])
    parser.add_argument("--split", nargs='?', default=DEFAULTS["split"])
    parser.add_argument("--batch_size", nargs='?', type=int, default=DEFAULTS["batch_size"])
    parser.add_argument("--max_length", nargs='?', type=int, default=DEFAULTS["max_length"])
    parser.add_argument("--torch_dtype", nargs='?', default=DEFAULTS["torch_dtype"])
    parser.add_argument("--devices", nargs='?', default=DEFAULTS["devices"])
    parser.add_argument("--output", nargs='?', default=DEFAULTS["output"])
    args = parser.parse_args()
    main(vars(args))