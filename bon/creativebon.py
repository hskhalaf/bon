import random
import pandas as pd
import torch
import csv
from vllm import LLM, SamplingParams
from datasets import load_dataset, Dataset

MODEL_NAME = "meta-llama/Llama-3.3-70B-Instruct"
model = LLM(model=MODEL_NAME, tensor_parallel_size=4, max_model_len = 4096)

def load_prompts(prompts, num_prompts=5):
    return random.sample(prompts, min(num_prompts, len(prompts)))

def sample_candidate_sentence(candidate_sentences, temperature=1.0):
    if not candidate_sentences:
        return ("", [])
    lengths = torch.tensor([length for (_, length) in candidate_sentences], dtype=torch.float)
    logits = -lengths / temperature
    probs = torch.nn.functional.softmax(logits, dim=0)
    sampled_index = torch.multinomial(probs, num_samples=1).item()
    return candidate_sentences[sampled_index]

def pick_sentence(sentences_lengths, soft):
    if not sentences_lengths:
        return ""
    if soft == 0:
        return min(sentences_lengths, key=lambda x: x[1])[0]
    return sample_candidate_sentence(sentences_lengths, temperature=soft)[0]

def build_multi_turn_messages(system_content, user_question, assistant_response=None):
    conversation = [{"role": "system", "content": system_content}]
    
    if assistant_response is None:
        conversation.append({"role": "user", "content": user_question})
    else:
        conversation.append({"role": "user", "content": user_question})
        conversation.append({"role": "assistant", "content": assistant_response})
        conversation.append({"role": "user", "content": "Please continue elaboration without repeating."})
    
    return conversation


def batch_generate_next_sentence(
    current_texts,
    soft,
    n_candidates=5,
    max_new_tokens=50,
    all_generated_sentences=None
):
    if all_generated_sentences is None:
        all_generated_sentences = [[] for _ in range(len(current_texts))]


    sampling_params = SamplingParams(
        temperature=0.7,
        top_k=50,
        top_p=0.9,
        max_tokens=max_new_tokens,
        repetition_penalty=1.05
    )

    prompts = []
    for i, text in enumerate(current_texts):
        system_content = (
            "You are a helpful and creative assistant that engages with the user's text in multiple turns. "
            "In each turn, you must generate one single complete sentence."
            "Every sentence must elaborate or continue the conversation from the previous turn, "
            "without repetition or restating the same sentence. "
            "Each sentence you respond with should not be too long, limit to under 25 words."
            
        )
        
        conversation = build_multi_turn_messages(
            system_content,
            text,
            None if not all_generated_sentences[i] else " ".join(all_generated_sentences[i])
        )
        
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in conversation])
        prompts.append(prompt)

    outputs = model.generate(prompts, sampling_params)
    
    chosen_sentences = []
    for i, output in enumerate(outputs):
        candidate_sentences = []
        for text in output.outputs[:n_candidates]:
            sentence = text.text.strip()
            idx = sentence.find('.')
            if idx != -1:
                sentence = sentence[:idx+1]
            candidate_sentences.append((sentence, len(sentence)))
        
        chosen = pick_sentence(candidate_sentences, soft)
        chosen_sentences.append(chosen)
    
    return chosen_sentences

def batch_shortest_path_generation(prompts, n_candidates=5, max_sentences=5, max_tokens_per_sentence=50, soft=0.7):
    current_texts = prompts[:]
    all_generated_sentences = [[] for _ in range(len(prompts))]
    for _ in range(max_sentences):
        next_sents = batch_generate_next_sentence(
            current_texts, soft, n_candidates, max_tokens_per_sentence, all_generated_sentences
        )
        got_new_sentence = False
        for i, sent in enumerate(next_sents):
            if sent.strip():
                got_new_sentence = True
                all_generated_sentences[i].append(sent)
                current_texts[i] += " " + sent
        if not got_new_sentence:
            break
    return current_texts, all_generated_sentences

def single_candidate_shortest_path(prompts, max_sentences=5, max_tokens_per_sentence=50):
    current_texts = prompts[:]
    all_sents = [[] for _ in range(len(prompts))]
    for _ in range(max_sentences):
        next_sents = batch_generate_next_sentence(
            current_texts, soft=0, n_candidates=1, max_new_tokens=max_tokens_per_sentence, all_generated_sentences=all_sents
        )
        got_new_sentence = False
        for i, sent in enumerate(next_sents):
            if sent.strip():
                got_new_sentence = True
                all_sents[i].append(sent)
                current_texts[i] += " " + sent
        if not got_new_sentence:
            break
    return current_texts, all_sents

def main():
    max_tokens_per_sentence = 35
    max_sentences = 5
    ds = load_dataset("Dampfinchen/Creative_Writing_Multiturn")
    prompts = [entry[1]["value"] for entry in ds["train"]["conversations"] if len(entry) > 1 and entry[1]["from"] == "human"]
    raw_prompts = load_prompts(prompts, num_prompts=200)
    
    n_values = [2, 4, 8, 12, 18, 24, 32]
    soft_values = [0.1, 0.35, 0.6, 0.85, 1.1, 1.35, 1.6, 1.85, 2.1]
    
    for n in n_values:
        for soft in soft_values:
            final_texts_sbon, all_sentences_sbon = batch_shortest_path_generation(
                raw_prompts, n, max_sentences, max_tokens_per_sentence, soft
            )
            final_texts_bon, all_sentences_bon = batch_shortest_path_generation(
                raw_prompts, n, max_sentences, max_tokens_per_sentence, soft=0
            )
            final_texts_single, all_sentences_single = single_candidate_shortest_path(
                raw_prompts, max_sentences, max_tokens_per_sentence
            )
            
            output_filename = f"output_{MODEL_NAME.replace('/', '_')}_{n}_{soft}.csv"
            with open(output_filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["prompt", "sBonResponse", "BoNResponse", "GreedyResponse"])
                for i, prompt_text in enumerate(raw_prompts):
                    writer.writerow([
                        prompt_text,
                        final_texts_sbon[i][final_texts_sbon[i].find('?') + 1:],
                        final_texts_bon[i][final_texts_bon[i].find('?') + 1:],
                        final_texts_single[i][final_texts_single[i].find('?') + 1:],
                    ])
            print(f"Saved: {output_filename}")

if __name__ == "__main__":
    main()
