from datasets import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import pickle
import os
from config import CACHE_PATH
from typing import List, Tuple, Optional

class ResponseGenerator:
    """Handles generation of responses from language models with caching support"""
    
    def __init__(
        self,
        model_name: str,
        dataset_name: str,
        device: str = 'cuda:0',
        batch_size: int = 64,
        max_length: int = 100,
        use_cache: bool = True,
        tokenizer_name: Optional[str] = None
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.device = device
        self.batch_size = batch_size
        self.max_length = max_length
        self.use_cache = use_cache
        self.tokenizer_name = tokenizer_name or model_name
        
        self.model_name_short = self._get_short_model_name()
        self.generation_name = f'{dataset_name}_{self.model_name_short}_{max_length}'
        self.cache_file = os.path.join(CACHE_PATH, f"{self.generation_name}.pkl")
        
    def _get_short_model_name(self) -> str:
        parts = self.model_name.split('/')
        return parts[-1] if len(parts[-1]) > 0 else parts[-2]
    
    def _format_prompts(self, prompts: List[str]) -> List[str]:
        return [
            f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{prompt}

### Response:
"""
            for prompt in prompts
        ]
    
    def _initialize_model_and_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(
            self.tokenizer_name,
            trust_remote_code=True,
            padding_side="left"
        )
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.bfloat16
        ).to(self.device)
        
        return model, tokenizer
    
    def _generate_batch(self, model, tokenizer, batch: List[str]) -> List[str]:
        inputs = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_length,
                min_p=0.1,
                temperature=0.2,
                pad_token_id=model.config.eos_token_id
            )
            
        new_tokens = outputs[:, inputs.input_ids.shape[1]:]
        batch_texts = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        return [text.replace("\n", " ") for text in batch_texts]
    
    def generate(self, prompts: List[str]) -> Tuple[List[str], str]:
        os.makedirs(os.path.dirname(self.cache_file), exist_ok=True)
        
        if os.path.exists(self.cache_file) and self.use_cache:
            print(f'Generation: {self.generation_name} found in cache, loading from cache')
            with open(self.cache_file, 'rb') as f:
                return pickle.load(f), self.generation_name
        print(f'Generation: {self.generation_name} {"invalidating cache and " if os.path.exists(self.cache_file) else ""}generating responses')
        
        model, tokenizer = self._initialize_model_and_tokenizer()
        formatted_prompts = self._format_prompts(prompts)
        dataloader = DataLoader(formatted_prompts, batch_size=self.batch_size, shuffle=False)
        
        all_responses = []
        for batch in tqdm(dataloader, desc="Generating responses"):
            batch_responses = self._generate_batch(model, tokenizer, batch)
            all_responses.extend(batch_responses)
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(all_responses, f)
        print(f'Generation: {self.generation_name} generated and saved to cache')
        
        del model
        torch.cuda.empty_cache()
        return all_responses, self.generation_name

def generate_responses(
    prompts: List[str],
    model_name: str,
    dataset_name: str,
    device: str = 'cuda:0',
    batch_size: int = 64,
    max_length: int = 100,
    use_cache: bool = True,
    tokenizer_name: Optional[str] = None
) -> Tuple[List[str], str]:

    generator = ResponseGenerator(
        model_name=model_name,
        dataset_name=dataset_name,
        device=device,
        batch_size=batch_size,
        max_length=max_length,
        use_cache=use_cache,
        tokenizer_name=tokenizer_name
    )
    return generator.generate(prompts)