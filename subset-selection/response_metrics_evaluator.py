import torch.multiprocessing as mp
from prometheus_eval.vllm import VLLM
from prometheus_eval import PrometheusEval
from prometheus_eval.prompts import ABSOLUTE_PROMPT, SCORE_RUBRIC_TEMPLATE
from transformers import AutoTokenizer, AutoModel
import torch
import evaluate
from tqdm import tqdm
import pickle
import os
import numpy as np
from config import CACHE_PATH
from typing import List, Dict, Union, Optional, Tuple

class MetricEvaluator:
    """Handles evaluation of model outputs using multiple metrics"""
    
    @staticmethod
    def evaluate_rouge(predictions: List[str], references: List[str], batch_size: int) -> float:
        rouge_metric = evaluate.load('rouge')
        scores = []

        for i in tqdm(range(0, len(predictions), batch_size), desc="Evaluating ROUGE"):
            batch_predictions = predictions[i:i+batch_size]
            batch_references = references[i:i+batch_size]
            batch_scores = rouge_metric.compute(
                predictions=batch_predictions, 
                references=batch_references
            )['rouge1']
            scores.append(batch_scores)
        return np.mean(scores).item()

    @staticmethod
    def evaluate_bge(
        predictions: List[str], 
        references: List[str], 
        device: str,
        batch_size: int
    ) -> float:
        embedding_tokenizer = AutoTokenizer.from_pretrained('BAAI/bge-large-en-v1.5')
        embedding_model = AutoModel.from_pretrained('BAAI/bge-large-en-v1.5')
        embedding_model.to(device)
        embedding_model.eval()

        if embedding_tokenizer.pad_token is None:
            embedding_tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            embedding_model.resize_token_embeddings(len(embedding_tokenizer))

        def compute_embeddings(texts: List[str]) -> torch.Tensor:
            encoded_input = embedding_tokenizer(
                texts, 
                padding=True, 
                truncation=True, 
                return_tensors='pt'
            ).to(device)
            with torch.no_grad():
                model_output = embedding_model(**encoded_input)
                sentence_embeddings = model_output[0][:, 0]
            return torch.nn.functional.normalize(sentence_embeddings, dim=1)

        metrics = []
        for i in tqdm(range(0, len(predictions), batch_size), desc='Computing BGE'):
            batch_preds = predictions[i:i+batch_size]
            batch_refs = references[i:i+batch_size]

            pred_embs = compute_embeddings(batch_preds)
            ref_embs = compute_embeddings(batch_refs)

            scores = (ref_embs * pred_embs).sum(dim=1).cpu().numpy()
            metrics.extend(scores)

        return np.mean(metrics).item()

    @staticmethod
    def evaluate_laj(
        predictions: List[str],
        prompts: List[str],
        references: List[str],
        return_individual: bool = False
    ) -> Union[float, np.ndarray]:

        model = VLLM(model="prometheus-eval/prometheus-7b-v2.0")
        judge = PrometheusEval(model=model, absolute_grade_template=ABSOLUTE_PROMPT)

        rubric_data = {
            "criteria": """Evaluate the model's ability to follow instructions and deliver a high-quality response across the following dimensions:
            1. **Instruction Following**: How accurately and fully does the model adhere to the given instruction?
            2. **Accuracy**: Is the information correct, reliable, and factually sound?
            3. **Relevance**: Does the response directly address the question or task without unnecessary information?
            4. **Completeness**: Does the response cover all essential aspects of the instruction or question?
            5. **Depth**: How thoroughly does the response explore the topic? Does it demonstrate insightful analysis where appropriate?
            6. **Clarity**: Is the response well-organized, easy to follow, and free from ambiguity or confusion?
            7. **Creativity**: Does the response offer original or innovative approaches where applicable?
            8. **Helpfulness**: Does the response effectively meet the user's needs and provide value in solving the problem or addressing the query?""",

            "score1_description": "The response fails to meet expectations across most or all criteria. It does not follow the instruction, contains significant errors or misinformation, lacks relevance, is incomplete or shallow, unclear, unoriginal, and unhelpful.",
            "score2_description": "The response shows major deficiencies across several criteria. It partially follows the instruction but includes significant inaccuracies, is often irrelevant, incomplete, or lacks depth, clarity, creativity, and helpfulness.",
            "score3_description": "The response is average, meeting some but not all criteria. It follows the instruction but may fall short in terms of accuracy, depth, relevance, or helpfulness. Improvements in clarity and insightfulness may be needed.",
            "score4_description": "The response is strong, performing well across most criteria. It follows the instruction closely, is mostly accurate and relevant, provides good depth, and is well-structured. Minor improvements could enhance clarity, creativity, or helpfulness.",
            "score5_description": "The response excels in all or nearly all criteria. It fully follows the instruction, is highly accurate, directly relevant, complete, and demonstrates depth and insight. The response is well-organized, creative where appropriate, and very helpful in addressing the user's needs.",
        }

        score_rubric = SCORE_RUBRIC_TEMPLATE.format(**rubric_data)

        feedback, score = judge.absolute_grade(
            instructions=prompts,
            responses=predictions,
            rubric=score_rubric,
            reference_answers=references
        )
        metrics = [x for x in score if x is not None]

        # Clean up resources
        del model
        del judge
        torch.cuda.empty_cache()

        return np.array(metrics) if return_individual else np.array(metrics).mean().item()

class MetricsComputer:
    """Handles computation and caching of evaluation metrics"""
    
    def __init__(
        self,
        predictions: List[str],
        prompts: List[str],
        references: List[str],
        generation_name: str,
        device: str = 'cuda:0',
        bs_bge: int = 512,
        bs_rouge: int = 4096,
        metrics: List[str] = ['bge', 'rouge', 'laj'],
        use_cache: bool = True
    ):
        self.predictions = predictions
        self.prompts = prompts
        self.references = references
        self.generation_name = generation_name
        self.device = device
        self.bs_bge = bs_bge
        self.bs_rouge = bs_rouge
        self.metrics = metrics
        self.use_cache = use_cache
        self.valid_metrics = {'bge', 'rouge', 'laj'}
        
    def _validate_metrics(self) -> None:
        invalid_metrics = set(self.metrics) - self.valid_metrics
        if invalid_metrics:
            raise ValueError(f"Invalid metrics: {invalid_metrics}")
    
    def _get_cache_path(self, metric: str) -> str:
        metric_name = f'{self.generation_name}_{metric}'
        return os.path.join(CACHE_PATH, f"{metric_name}.pkl")
    
    def _load_from_cache(self, cache_file: str) -> Optional[float]:
        if os.path.exists(cache_file) and self.use_cache:
            print(f'Evaluate: {os.path.basename(cache_file)} found in cache')
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_file: str, value: float) -> None:
        os.makedirs(os.path.dirname(cache_file), exist_ok=True)
        with open(cache_file, 'wb') as f:
            pickle.dump(value, f)
    
    def compute(self) -> Dict[str, float]:
        self._validate_metrics()
        results = {}
        
        for metric in self.metrics:
            cache_file = self._get_cache_path(metric)
            value = self._load_from_cache(cache_file)
            
            if value is None:
                print(f'Evaluate: Computing {metric} {"(invalidating cache)" if os.path.exists(cache_file) else ""}')
                
                if metric == 'rouge':
                    value = MetricEvaluator.evaluate_rouge(
                        self.predictions, self.references, self.bs_rouge
                    )
                elif metric == 'bge':
                    value = MetricEvaluator.evaluate_bge(
                        self.predictions, self.references, self.device, self.bs_bge
                    )
                elif metric == 'laj':
                    value = MetricEvaluator.evaluate_laj(
                        self.predictions, self.prompts, self.references
                    )
                
                self._save_to_cache(cache_file, value)
                print(f'Evaluate: {metric} computed and cached')
            
            results[metric] = value
            
        return results

def compute_metrics(
    predictions: List[str],
    prompts: List[str],
    references: List[str],
    generation_name: str,
    device: str = 'cuda:0',
    bs_bge: int = 512,
    bs_rouge: int = 4096,
    metrics: List[str] = ['bge', 'rouge'],
    use_cache: bool = True
) -> Dict[str, float]:

    computer = MetricsComputer(
        predictions=predictions,
        prompts=prompts,
        references=references,
        generation_name=generation_name,
        device=device,
        bs_bge=bs_bge,
        bs_rouge=bs_rouge,
        metrics=metrics,
        use_cache=use_cache
    )
    return computer.compute()