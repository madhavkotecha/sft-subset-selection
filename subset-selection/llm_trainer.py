import os
import random
from datetime import datetime
from config import CACHE_PATH
import evaluate
import pytz
import torch
from datasets import Dataset
import numpy as np
from peft import LoraConfig
from tqdm import tqdm
from transformers import (
    BitsAndBytesConfig, 
    AutoModelForCausalLM, 
    AutoTokenizer, 
    EvalPrediction
)
from trl import SFTTrainer, SFTConfig

class ModelTrainer:
    """Handles model fine-tuning with LoRA and quantization"""
    
    def __init__(
        self,
        base_model_id: str,
        subset_name: str,
        use_cache: bool = True,
        tag: str = None
    ):
        self.base_model_id = base_model_id
        self.subset_name = subset_name
        self.use_cache = use_cache
        self.tag = tag
        self.model_name = f"{base_model_id.split('/')[-1]}_{subset_name}"
        self.model_dir = os.path.join(CACHE_PATH, self.model_name)
        
    def _setup_quantization_config(self):
        quant_storage_dtype = torch.bfloat16
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_storage=quant_storage_dtype,
        ), quant_storage_dtype
    
    def _load_model_and_tokenizer(self, quant_config, dtype):
        model = AutoModelForCausalLM.from_pretrained(
            self.base_model_id,
            quantization_config=quant_config,
            trust_remote_code=True,
            attn_implementation="flash_attention_2",
            torch_dtype=dtype,
            use_cache=False,
            device_map='auto'
        )
        
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        # Handle multi-GPU setup
        if torch.cuda.device_count() > 1:
            print(f"----------using {torch.cuda.device_count()}*GPUs----------")
            model = torch.nn.DataParallel(model).module
            
        tokenizer = AutoTokenizer.from_pretrained(self.base_model_id)
        tokenizer.pad_token = tokenizer.eos_token
        
        return model, tokenizer
    
    def _prepare_datasets(self, prompts, references, prompts_val, references_val):
        def format_prompt(prompt, reference):
            return f"""Below is an instruction that describes a task. Write a response that appropriately completes the request.

{prompt}

### Response:
{reference}
            """
        
        train_texts = [format_prompt(p, r) for p, r in zip(prompts, references)]
        val_texts = [format_prompt(p, r) for p, r in zip(prompts_val, references_val)]
        
        return (
            Dataset.from_dict({"text": train_texts}),
            Dataset.from_dict({"text": val_texts})
        )
    
    def _get_training_config(self, epochs, train_bs, eval_bs, eval_strategy, save_strategy):
        return SFTConfig(
            max_seq_length=1024,
            packing=True,
            eval_packing=False,
            dataset_text_field="text",
            dataset_kwargs={
                "add_special_tokens": False,
                "append_concat_token": False,
            },
            output_dir=self.model_dir,
            num_train_epochs=epochs,
            per_device_train_batch_size=train_bs,
            per_device_eval_batch_size=eval_bs,
            gradient_accumulation_steps=1,
            eval_accumulation_steps=1,
            eval_strategy=eval_strategy,
            save_strategy=save_strategy,
            save_total_limit=1,
            learning_rate=2.5e-5,
            bf16=True,
            run_name=self._get_run_name(),
            logging_steps=10,
            optim="paged_adamw_8bit",
            lr_scheduler_type="constant",
            weight_decay=0.01,
            report_to="tensorboard",
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={'use_reentrant': True},
        )
    
    def _get_run_name(self):
        current_time = datetime.now(pytz.timezone('Asia/Kolkata'))
        formatted_time = current_time.strftime('%d-%m-%Y %H:%M:%S')
        return f'{formatted_time}' if self.tag is None else f'{self.tag} | {formatted_time}'
    
    def fine_tune(
        self,
        prompts,
        references,
        prompts_val,
        references_val,
        epochs=3,
        train_bs=24,
        eval_bs=24,
        eval_strategy="no",
        save_strategy="no"
    ):
        adapter_config_path = os.path.join(self.model_dir, 'adapter_config.json')
        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)

        if os.path.exists(adapter_config_path) and self.use_cache:
            print(f"Finetune: Fine-tuned model {self.model_name} found in cache.")
            return self.model_dir
            
        print(f"Finetune: Training model {self.model_name}")
        
        # Setup model components
        quant_config, dtype = self._setup_quantization_config()
        model, tokenizer = self._load_model_and_tokenizer(quant_config, dtype)
        train_dataset, valid_dataset = self._prepare_datasets(
            prompts, references, prompts_val, references_val
        )
        
        # Configure LoRA
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules="all-linear",
            bias="none",
            lora_dropout=0.1,
            task_type="CAUSAL_LM",
        )
        
        # Setup trainer
        trainer = SFTTrainer(
            model=model,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            peft_config=peft_config,
            compute_metrics=self._compute_metrics(tokenizer),
            args=self._get_training_config(
                epochs, train_bs, eval_bs, eval_strategy, save_strategy
            ),
        )
        
        if trainer.accelerator.is_main_process:
            trainer.model.print_trainable_parameters()
        
        # Train and save
        trainer.train()
        if hasattr(trainer, 'is_fsdp_enabled') and trainer.is_fsdp_enabled:
            trainer.accelerator.state.fsdp_plugin.set_state_dict_type("FULL_STATE_DICT")
        trainer.save_model()
        
        print(f"Finetune: Model fine-tuning completed for {self.model_name} and saved to cache.")
        
        # Cleanup
        del model
        torch.cuda.empty_cache()
        return self.model_dir
    
    @staticmethod
    def _compute_metrics(tokenizer):
        def compute_fn(p: EvalPrediction):
            logits, labels = p
            predictions = tokenizer.batch_decode(
                np.argmax(logits, axis=-1),
                skip_special_tokens=True
            )
            labels[labels < 0] = tokenizer.eos_token_id
            references = tokenizer.batch_decode(labels, skip_special_tokens=True)
            
            rouge = evaluate.load('rouge')
            rouge_scores = rouge.compute(predictions=predictions, references=references)
            return {'rouge1': rouge_scores['rouge1']}
        return compute_fn

def finetune_model(
    base_model_id,
    prompts,
    references,
    prompts_val,
    references_val,
    subset_name,
    use_cache=True,
    tag=None,
    epochs=3,
    train_bs=24,
    eval_bs=24,
    eval_strategy="no",
    save_strategy="no"
):
    trainer = ModelTrainer(base_model_id, subset_name, use_cache, tag)
    return trainer.fine_tune(
        prompts,
        references,
        prompts_val,
        references_val,
        epochs,
        train_bs,
        eval_bs,
        eval_strategy,
        save_strategy
    )