import random
import argparse
import pytz
import json
import os
from datetime import datetime
from config import CACHE_PATH
from typing import Dict, Any, Tuple, List

from llm_trainer import finetune_model
from llm_inference import generate_responses
from response_metrics_evaluator import compute_metrics
from load_dataset import load_gsm8k, load_mixInstruct, load_mmlu
from delift.utility import get_utility
from subset_selector import create_subset, get_subset

class ExperimentConfig:
    """Configuration handler for LLM fine-tuning experiments"""
    
    def __init__(self):
        self.parser = self._setup_argument_parser()
        
    def _setup_argument_parser(self) -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(description="Fine-tune and evaluate an LLM model.")
        
        # Dataset selection and related parameters
        parser.add_argument("--dataset-name", type=str,
                            choices=['mix-instruct', 'gsm8k', 'mmlu'],
                            default='mix-instruct',
                            help="Dataset name")
        parser.add_argument("--train_seed", type=int, default=42, help="Random seed for training data selection")
        parser.add_argument("--train_length", type=int, default=21000, help="Number of training samples")
        parser.add_argument("--val_seed", type=int, default=42, help="Random seed for validation data selection")
        parser.add_argument("--val_length", type=int, default=1000, help="Number of validation samples")
        
        # Subset method selection parameters
        parser.add_argument("--method", type=str, 
                          choices=['initial', 'random', 'delift-se', 'full'], default='initial', help="Subset selection method")
        parser.add_argument("--subset_size", type=float, default=0.3, help="Proportion of data to keep in subset selection")
        parser.add_argument("--random_seed", type=int, default=42, help="Seed for random subset selection")
        
        # Model name and parameters
        parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-3B', help="Base model identifier")
        parser.add_argument("--epochs", type=int, default=3, help="Number of fine-tuning epochs")
        parser.add_argument("--generation_max_length", type=int, default=150, help="Maximum generation length during inference")
        parser.add_argument("--generation_batch_size", type=int, default=64, help="Batch size during inference")
        parser.add_argument("--tag", type=str, default=None, help="Tag name to uniquely identify experiments")
        
        return parser
    
    def parse_args(self) -> argparse.Namespace:
        return self.parser.parse_args()

class ExperimentRunner:
    """Handles the execution of LLM fine-tuning experiments"""
    
    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.experiment_name = self._generate_experiment_name()
        
    def _generate_experiment_name(self) -> str:
        formatted_time = datetime.now(pytz.timezone('Asia/Kolkata')).strftime('%d-%m-%Y-%H:%M:%S')
        name = f'{self.args.model}_{self.args.method}_{formatted_time}'
        if self.args.tag:
            name += self.args.tag
        return name
    
    def _print_parameters(self) -> None:
        print(f'\nRunning experiment "{self.experiment_name}" with parameters:')
        max_key_length = max(len(str(k)) for k in vars(self.args).keys())
        for k, v in vars(self.args).items():
            print(f"{k:<{max_key_length}} : {v}")
        print("")
    
    def _load_datasets(self) -> Tuple[List[str], List[str], str, List[str], List[str], str]:
        if self.args.dataset_name == 'gsm8k':
            train_data = load_gsm8k("train", self.args.train_length, seed=self.args.train_seed)
            val_data = load_gsm8k("validation", self.args.val_length, seed=self.args.val_seed)
        elif self.args.dataset_name == 'mix-instruct':
            train_data = load_mixInstruct("train", self.args.train_length, seed=self.args.train_seed)
            val_data = load_mixInstruct("validation", self.args.val_length, seed=self.args.val_seed)
        elif self.args.dataset_name == 'mmlu':
            train_data = load_mmlu("train", self.args.train_length, seed=self.args.train_seed)
            val_data = load_mmlu("validation", self.args.val_length, seed=self.args.val_seed)
        
        return (*train_data, *val_data)
    
    def _apply_subset_selection(self, prompts: List[str], references: List[str], ds_name: str) -> Tuple[List[str], List[str], str]:
        subset_name = ds_name
        
        if self.args.method == 'delift-se':
            utility, utility_name = get_utility(prompts, references, ds_name)
            subset, subset_name = create_subset(utility, utility_name, k=self.args.subset_size)
            prompts, references = get_subset(subset, prompts, references)
        elif self.args.method == 'random':
            subset_name = f'{ds_name}_random_{self.args.subset_size}'
            subset_size = int(self.args.subset_size * len(prompts))
            subset_indices = random.Random(self.args.random_seed).sample(range(len(prompts)), subset_size)
            prompts = [prompts[i] for i in subset_indices]
            references = [references[i] for i in subset_indices]
            
        return prompts, references, subset_name
    
    def _save_results(self, metrics: Dict[str, float]) -> None:
        results = {
            "experiment_name": self.experiment_name,
            "parameters": vars(self.args),
            "metrics": metrics
        }
        
        log_file = os.path.join(CACHE_PATH, f'{self.experiment_name}.txt')
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        with open(log_file, "w") as f:
            json.dump(results, f, indent=4)
    
    def _print_results(self, metrics: Dict[str, float]) -> None:
        print(f"\nResults for the experiment: {self.experiment_name}")
        max_key_length = max(len(str(k)) for k in metrics.keys())
        for k, v in metrics.items():
            print(f"{k:<{max_key_length}} : {v}")
    
    def run(self) -> None:
        self._print_parameters()
        
        prompts, references, ds_name, prompts_val, references_val, ds_name_val = self._load_datasets()
        prompts, references, subset_name = self._apply_subset_selection(prompts, references, ds_name)
        
        model_dir = self.args.model
        if self.args.method != 'initial':
            model_dir = finetune_model(
                self.args.model, prompts, references,
                prompts_val, references_val, subset_name,
                epochs=self.args.epochs, tag=self.args.tag
            )
        
        responses, generation_name = generate_responses(
            prompts_val, model_dir, ds_name_val,
            max_length=self.args.generation_max_length,
            batch_size=self.args.generation_batch_size
        )
        metrics = compute_metrics(responses, prompts_val, references_val, generation_name)
        
        self._save_results(metrics)
        self._print_results(metrics)

if __name__ == "__main__":
    config = ExperimentConfig()
    runner = ExperimentRunner(config.parse_args())
    runner.run()