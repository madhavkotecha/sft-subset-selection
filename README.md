# sft-subset-selection

There are two different approach we followed

1. subset-selection
    To run code:

        python3 manage.py [parameters]
        Alternatively, you can use run.sh to run the multiple subset-selection parallely across multiple GPUs.

        Parameters
            --dataset-name
            Type: str
            Choices: mix-instruct, gsm8k, mmlu
            Default: mix-instruct
            Description: Dataset to use for training and evaluation.

            --train_seed
            Type: int
            Default: 42
            Description: Random seed for training data selection.

            --train_length
            Type: int
            Default: 21000
            Description: Number of training samples.

            --val_seed
            Type: int
            Default: 42
            Description: Random seed for validation data selection.

            --val_length
            Type: int
            Default: 1000
            Description: Number of validation samples.

            --method
            Type: str
            Choices: initial, random, delift-se, full
            Default: initial
            Description: Subset selection method.

            --subset_size
            Type: float
            Default: 0.3
            Description: Proportion of data to keep in subset selection.

            --random_seed
            Type: int
            Default: 42
            Description: Seed for random subset selection.

            --model
            Type: str
            Default: meta-llama/Llama-3.2-3B
            Description: Base model identifier.

            --epochs
            Type: int
            Default: 3
            Description: Number of fine-tuning epochs.

            --generation_max_length
            Type: int
            Default: 150
            Description: Maximum generation length during inference.

            --generation_batch_size
            Type: int
            Default: 64
            Description: Batch size during inference.

            --tag
            Type: str
            Default: None
            Description: Tag name to uniquely identify experiments.


2. math_subset-selection

    ├── EDA/                           # Exploratory Data Analysis
    │   └── diversity_utility_score.py  # Code to generate diversity matrices for GSM8K samples
    │
    ├── Baselines/                     # Baseline subset selection methods
    │   ├── baselines.py           # Determinantal Point Process selection implementation       # Random subset selection implementation
    │
    ├── full_fine_tuning/              # Full dataset fine-tuning
    │   ├── full_fine_tuning.py                   # Training code for full GSM8K dataset              # Evaluation code for full dataset models
    │
    ├── perplexity_diversity_score_selected_subsets/  # Our method without CoT loss
    │   ├── fine_tuning_subset_selection_perplexity.py         # Main implementation of our perplexity-based selection        # Code for calculating diversity scores
    │
    ├── subset_fine_tuning/            # Fine-tuning on selected subsets
    │   ├── fine_tuning_900_1900.py                   # Training code for subset fine-tuning               # Evaluation code for subset models
    │
    └── cot_our_method.py                # Our method with Chain-of-Thought loss   # Implementation for selecting top 20% samples using CoT loss             # Code for calculating CoT-based scores
