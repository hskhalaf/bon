from datasets import load_dataset
import pandas as pd

def load_huggingface(dataset_name=None, split=None, **kwargs):
    if dataset_name is None:
        raise ValueError("dataset_name must be specified!")
    
    dataset = load_dataset(dataset_name)
    data = process_default(dataset, split=split)
    return data


def prompt(text):
    last_assistant = text.rfind("Assistant:")
    return text[:last_assistant].strip()


def answer(text):
    last_assistant = text.rfind("Assistant:")
    return text[last_assistant:].strip()


def process_default(dataset, split=None):
    df = pd.DataFrame({
        "prompt": [prompt(text) for text in dataset[split]['chosen']],
        "chosen": [answer(text) for text in dataset[split]['chosen']],
        "rejected": [answer(text) for text in dataset[split]['rejected']],
    })
    df["row_id"] = df.index.values
    return df