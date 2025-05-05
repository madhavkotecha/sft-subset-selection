import sys
import os
import pickle
import torch

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel
from load_dataset import load_mixInstruct
from config import CACHE_PATH

class TextEncoder:
    
    def __init__(self, embedding_model_name='BAAI/bge-large-en-v1.5'):
        self.embedding_model_name = embedding_model_name
        self.max_length = 512
        
    def _setup_model(self):
        tokenizer = AutoTokenizer.from_pretrained(self.embedding_model_name)
        tokenizer.max_subtokens_sequence_length = self.max_length
        tokenizer.model_max_length = self.max_length
        
        model = AutoModel.from_pretrained(self.embedding_model_name).to('cuda')
        model.eval()
        return tokenizer, model
        
    def encode_batch(self, prompts, references, batch_size=16):
        tokenizer, model = self._setup_model()
        
        data_to_encode = [f"{prompt} {reference}" for prompt, reference in zip(prompts, references)]
        encoded_input = tokenizer(
            data_to_encode,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(model.device)
        
        output = []
        for i in tqdm(range(0, len(encoded_input['input_ids']), batch_size), 
                     desc='Encoding data to vectors'):
            batch_output = model(
                input_ids=encoded_input['input_ids'][i:i+batch_size],
                attention_mask=encoded_input['attention_mask'][i:i+batch_size]
            ).pooler_output.detach()
            output.extend(batch_output)
            
        output = torch.stack(output)
        
        del tokenizer, model
        return output

def get_utility_enc(prompts, references, dataset_name, embedding_model_name='BAAI/bge-large-en-v1.5'):
    
    utility_name = f'{dataset_name}_encodes'
    cache_file = os.path.join(CACHE_PATH, f"{utility_name}.pkl")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f'Utility: {utility_name} found in cache')
        with open(cache_file, 'rb') as f:
            return pickle.load(f), utility_name
    print(f'Utility: {utility_name} not found in cache')
    
    encoder = TextEncoder(embedding_model_name)
    utility = encoder.encode_batch(prompts, references).to("cpu")
    utility = np.array(utility)
    with open(cache_file, 'wb') as f:
        pickle.dump(utility, f)
    print(f'Utility: {utility_name} computed and saved')
    return utility, utility_name

def encode(prompts, references, embedding_model_name):
    tokenizer = AutoTokenizer.from_pretrained(embedding_model_name)
    tokenizer.max_subtokens_sequence_length = 512
    tokenizer.model_max_length = 512
    model = AutoModel.from_pretrained(embedding_model_name).to('cuda')
    model.eval()

    data_to_encode = [p + " " + r for p, r in zip(prompts, references)]
    encoded_input = tokenizer(data_to_encode, padding=True, truncation=True, return_tensors='pt').to(model.device)
    input_ids = encoded_input['input_ids']
    attention_mask = encoded_input['attention_mask']

    bs = 16
    output = []
    for i in tqdm(range(0, len(encoded_input['input_ids']), bs), desc='Encoding data to vectors'):
        output.extend(model(input_ids=input_ids[i:i+bs], attention_mask=attention_mask[i:i+bs]).pooler_output.detach())

    output = torch.stack(output)
    del tokenizer, model
    return output