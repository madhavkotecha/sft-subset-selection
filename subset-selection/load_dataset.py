import os
import pickle
from datasets import load_dataset
import pandas as pd
from config import CACHE_PATH

def load_gsm8k(split, max_length=None, seed=42):
    def generate_prompt(question):
        return f"{question}\nAnswer:"

    dataset_name = f"gsm8k_{split}_{max_length}_{seed}"
    cache_file = os.path.join(CACHE_PATH, f"{dataset_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    prompts, references = [], []
    if os.path.exists(cache_file):
        print(f'Dataset: {dataset_name} found in cache')
        with open(cache_file, 'rb') as f:
            prompts, references = pickle.load(f)
    else:
        print(f'Dataset: {dataset_name} not found in cache')
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        max_length = max_length if max_length is not None else len(dataset)
        dataset = dataset.shuffle(seed=seed).select(range(max_length))
        prompts = [generate_prompt(example['question']) for example in dataset]
        references = [example['answer'] for example in dataset]
        with open(cache_file, 'wb') as f:
            pickle.dump((prompts, references), f)
    
    print(f'Dataset: {dataset_name} loaded')
    return prompts, references, dataset_name


def load_mmlu(split, max_length=None, seed=42):
    def generate_prompt(question, choices):
        choices_str = "\n".join([f"{chr(65+i)}. {choice}" for i, choice in enumerate(choices)])
        return f"{question}\n{choices_str}\nAnswer:"

    dataset_name = f"mmlu_{split}_{max_length}_{seed}"
    cache_file = os.path.join(CACHE_PATH, f"{dataset_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    prompts, references = [], []
    if os.path.exists(cache_file):
        print(f'Dataset: {dataset_name} found in cache')
        with open(cache_file, 'rb') as f:
            prompts, references = pickle.load(f)
    else:
        print(f'Dataset: {dataset_name} not found in cache')
        dataset = load_dataset("cais/mmlu", 'all', split=split)
        max_length = max_length if max_length is not None else len(dataset)
        dataset = dataset.shuffle(seed=42).select(range(max_length))
        prompts = [generate_prompt(example['question'], example['choices']) for example in dataset]
        references = [chr(65 + example['answer']) for example in dataset]
        with open(cache_file, 'wb') as f:
            pickle.dump((prompts, references), f)
    
    print(f'Dataset: {dataset_name} loaded')
    return prompts, references, dataset_name


def load_mixInstruct(split, max_length, seed=42):
    dataset_name = f"mix-instruct_{split}_{max_length}_{seed}"
    cache_file = os.path.join(CACHE_PATH, f"{dataset_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    if os.path.exists(cache_file):
        print(f'Dataset: {dataset_name} found in cache')
        dataset = pd.read_pickle(cache_file)
    else:
        print(f'Dataset: {dataset_name} not found in cache')
        dataset = load_dataset("llm-blender/mix-instruct")[split].shuffle(seed=seed).to_pandas()[:max_length]
        dataset.to_pickle(cache_file)
    
    print(f'Dataset: {dataset_name} loaded')
    prompts = dataset['instruction'] + "\n" + dataset['input']
    references = dataset['output']
    return list(prompts), list(references), dataset_name