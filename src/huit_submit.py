import pandas as pd
import os
import yaml
import numpy as np
from tqdm import tqdm
import aiohttp
import asyncio
import json

URL = "https://go.apis.huit.harvard.edu/ais-openai-direct-limited-schools/v1/chat/completions"

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
SECRETS_PATH = os.path.join(ROOT_DIR, 'secrets.yaml')
SECRETS = yaml.safe_load(open(SECRETS_PATH))
HEADERS = {
    'Content-Type': 'application/json',
    'api-key': 'MPSA9cA0dTEMJmGmiE1g2ibvtDXI5uuPQZywRrlPTHRKFAux'
}

CONFIG = {
    # "logprobs": True,
    # "top_logprobs": 20,
    "temperature": 0.7,
    "max_tokens": 1
}

PARALLEL_SIZE = 20
MAX_RETRIES = 5


async def post_request(session, payload):
    attempts = 0
    delay = 2
    response = None
    while attempts < MAX_RETRIES:
        try:
            async with session.post(URL, headers=HEADERS, data=json.dumps(payload)) as response:
                response = await response.json(content_type=None)
                return response

        except asyncio.TimeoutError:
            print("The request has timed out.")
        except Exception as e:
            attempts += 1
            print("Request failed: ", str(e))
            if attempts < MAX_RETRIES:
                await asyncio.sleep(delay) 
                delay *= 2 
            return {'content': [{'text': None}]}
        
    if response is None:
        raise Exception("All retry attempts failed.")


async def post_many_requests(prompts_df, config, existing_result_df=None):
    model_id = config['model_id'].split("/")[-1]
    requested_tokens = ["A", "B", "NA"]

    full_score_df = existing_result_df
    async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(ssl=False)) as session:
        tasks = []
        for _, row in prompts_df.iterrows():
            payload = {
                'model': model_id,
                'messages': 
                    row['prompt']
            } | CONFIG
            tasks.append(post_request(session, payload))

        with tqdm(total=len(tasks), desc="Processing") as pbar:
            for i in range(0, len(tasks), PARALLEL_SIZE):
                end = i + PARALLEL_SIZE if i + PARALLEL_SIZE < len(tasks) else len(tasks)
                tasks_parallel = tasks[i:end]
                responses = await asyncio.gather(*tasks_parallel)
                pbar.update(len(tasks_parallel))

                token_probs = {f'prob_{t}': [] for t in requested_tokens}
                for response in responses:
                    try:
                        generated_token = response['choices'][0]['message']['content']
                    except KeyError:
                        generated_token = None
                        
                    if generated_token is not None:
                        for requested_token in requested_tokens:
                            token_probs[f'prob_{requested_token}'].append(1.0 if generated_token == requested_token else 0.0)
                    else:
                        for requested_token in requested_tokens:
                            token_probs[f'prob_{requested_token}'].append(0.0)

                scores_df = pd.DataFrame(token_probs | {
                    'row_id': prompts_df['row_id'].iloc[i:end],
                    'attribute_id': prompts_df['attribute_id'].iloc[i:end],
                })
                if full_score_df is None:
                    scores_df.to_csv(config['output'], index=False)
                    full_score_df = scores_df
                else:
                    scores_df.to_csv(config['output'], mode='a', index=False, header=False)
                    full_score_df = pd.concat([full_score_df, scores_df], ignore_index=True)
    return full_score_df


def annotate_prompts(prompts_df, config, existing_result_df=None):
    print(f"Processing {len(prompts_df)} prompts with model {config['model_id']}.")
    full_score_df = asyncio.run(post_many_requests(prompts_df, config, existing_result_df))
    return full_score_df
