"""Masked Pretraining of DNA Sequence Language Models

This code is a training tool for DNA sequence language models developed based on the Hugging Face Transformers library,
supporting two training modes: Masked Language Model (MLM) and Causal Language Model (CLM).
The code is primarily adapted from Hugging Face's official examples,
with specific optimizations for modifying tokenizers originally designed for natural language processing to better match DNA sequence data training requirements.

Main References:
    - huggingface transformers(https://github.com/huggingface/transformers)
    - DNABERT (https://github.com/jerryji1993/DNABERT)
    - nucleotide-transformer(https://github.com/instadeepai/nucleotide-transformer)
    - Plant_DNA_LLMs (https://github.com/zhangtaolab/Plant_DNA_LLMs)

Example Command:
    python 01-pretrain.py  \
        --model_name_or_path  ./model_local_path \
        --train_data ./pretrain_data.txt \
        --output_dir ./output

Code Information:
    Author: YangLin
    Team: Xia_lab
    Date: 2025.11.20
"""
import os
import torch
import numpy as np
import random
from dataclasses import dataclass, field
from typing import Optional
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForMaskedLM
from datasets import load_dataset
from transformers import DataCollatorForLanguageModeling
from transformers import Trainer, TrainingArguments, HfArgumentParser
from transformers import AutoConfig
#from transformers import AutoModel


## Arguments definition ###############################################
@dataclass
class ModelArguments:
    model_name_or_path: str = field(default=None)           # Model name or path
    tokenizer_path: Optional[str] = field(default=None)     # Tokenizer path
    load_checkpoint: Optional[str] = field(default=None)    # Checkpoint loading path
    is_mlm: bool = field(default=True, metadata={"help": "Is masked language model."})      # Whether it is a masked language model


@dataclass
class DataArguments:
    train_data: str = field(default=None,metadata={"help": "Path to the training data."})   # Path to training data
    eval_data: Optional[str] = field(default=None, metadata={"help": "Path to the validation data."})  # Path to validation data
    test_data: Optional[str] = field(default=None, metadata={"help": "Path to the test data."})        # Path to test data
    split: float = field(default=0, metadata={"help": "Test split"})  # Test set split ratio
    shuffle: bool = field(default=True)  # Whether to shuffle data when splitting test set


@dataclass
class TrainingArguments(TrainingArguments):
    cache_dir: Optional[str] = field(default=None)  # Cache directory
    run_name: str = field(default="runs")           # Run name
    optim: str = field(default="adamw_torch")       # Optimizer
    seed: int = field(default=12)                   # Random seed
    model_max_length: int = field(default=512, metadata={"help": "Maximum sequence length."})  # Maximum input sequence length for the model
    gradient_accumulation_steps: int = field(default=4)     # Gradient accumulation steps
    per_device_train_batch_size: int = field(default=16)    # Training batch size per device
    per_device_eval_batch_size: int = field(default=16)     # Evaluation batch size per device
    num_train_epochs: int = field(default=5)                # Number of training epochs
    fp16: bool = field(default=False)  # Whether to enable mixed precision training (16-bit floating point)
    bf16: bool = field(default=False)  # Whether to enable mixed precision training (bfloat16)
    logging_steps: Optional[int] = field(default=50)    # Logging interval steps
    logging_strategy: str = field(default='steps')      # Logging strategy
    save_steps: Optional[int] = field(default=1)    # Model saving interval steps
    save_strategy: str = field(default='epoch')     # Model saving strategy
    warmup_ratio: float = field(default=0.05)       # Learning rate warmup ratio
    weight_decay: float = field(default=0.01)       # Weight decay
    learning_rate: float = field(default=1e-5)      # Initial learning rate
    adam_beta1: float = field(default=0.9)          # Adam optimizer beta1 parameter
    adam_beta2: float = field(default=0.98)         # Adam optimizer beta2 parameter
    adam_epsilon: float = field(default=1e-6)       # Adam optimizer epsilon parameter
    save_total_limit: int = field(default=100)      # Maximum number of saved models
    output_dir: str = field(default="output")       # Output directory
    dataloader_pin_memory: bool = field(default=False)  # Whether to pin data in memory
    find_unused_parameters: bool = field(default=False) # Whether to check for unused parameters
    checkpointing: bool = field(default=False)          # Whether to enable checkpoint saving
    no_safe_serialization: bool = field(default=False)  # Whether to disable safe serialization


#### train ##############################################
def train(args, model, trainset, validset, data_collator, output_dir, resume_from_checkpoint=None):
    # Initialize Trainer
    trainer = Trainer(
        args=args,
        model=model,
        train_dataset=trainset,
        eval_dataset=validset,
        data_collator=data_collator,
    )
    # Continue training from checkpoint or train directly
    if resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    else:
        trainer.train()
    # Save the trained model
    trainer.save_model(output_dir)

def main():
    # Parse command line arguments
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    print("Model Arguments:", model_args)
    print("Data Arguments:", data_args)
    print("Training Arguments:", training_args)

    seed = training_args.seed
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    output_dir = training_args.output_dir
    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    print("Tokenizer loading from:", model_args.model_name_or_path)
    print("Tokenizer loaded:", tokenizer)

    if model_args.is_mlm:
        # Masked language model
        # Initialize empty model from config (without loading weights)
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True,  cache_dir=training_args.cache_dir)
        model = AutoModelForMaskedLM.from_config(config)

        # Alternative model loading methods
        # model = AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)   # Load pre-trained weights
        # model = AutoModelForMaskedLM.from_config(AutoModelForMaskedLM.from_pretrained(model_args.model_name_or_path).config) # Initialize model based on config

    else:
        # Causal language model
        # Initialize empty model from config (without loading weights)
        config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True, cache_dir=training_args.cache_dir)
        model = AutoModelForCausalLM.from_config(config)

        # Alternative model loading methods
        # model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)   # Load pre-trained weights
        # model = AutoModelForCausalLM.from_config(AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path).config) # Initialize model based on config

    print("Model loading from:", model_args.model_name_or_path)
    print("Model loaded:", model)
    print("Output directory:", output_dir)

    def sync_special_tokens(model, tokenizer):
        """Synchronize special tokens between model and tokenizer, handling vocabulary changes"""
        # 1. First print existing special token information in model and tokenizer
        print("\n=== Original  ===")
        print("\n--- Special tokens of model.config ---")
        for attr in dir(model.config):
            if attr.endswith('_token_id') and not attr.startswith('__'):
                print(f"{attr}: {getattr(model.config, attr)}")

        print("\n--- Special tokens of Tokenizer ---")
        special_tokens = [
            'bos_token', 'eos_token', 'unk_token', 'sep_token',
            'cls_token', 'pad_token', 'mask_token'
        ]

        for token_type in special_tokens:
            token = getattr(tokenizer, token_type, None)
            token_id = getattr(tokenizer, f"{token_type}_id", None)
            if token is not None and token_id is None:
                token_id = tokenizer.convert_tokens_to_ids(token)
                setattr(tokenizer, f"{token_type}_id", token_id)
            print(f"{token_type}: '{token}' (ID: {token_id})")

        print("\n=== Actioning synchronization ===")

        # 2. Synchronize special tokens
        for token_type in special_tokens:
            # Get token and ID from tokenizer
            tokenizer_token = getattr(tokenizer, token_type, None)
            tokenizer_id = getattr(tokenizer, f"{token_type}_id", None)

            if tokenizer_token is not None and tokenizer_id is None:
                tokenizer_id = tokenizer.convert_tokens_to_ids(tokenizer_token)
                setattr(tokenizer, f"{token_type}_id", tokenizer_id)

            # Get ID from model config
            config_attr = f"{token_type}_id"
            model_id = getattr(model.config, config_attr, None)

            # Synchronization logic
            if tokenizer_token is not None:
                # If model doesn't have this config but tokenizer does, add to model config
                if not hasattr(model.config, config_attr):
                    print(f"Adding {config_attr} to model.config: {tokenizer_id}")
                    setattr(model.config, config_attr, tokenizer_id)
                # If IDs are inconsistent, update model to match tokenizer
                elif model_id != tokenizer_id:
                    print(f"Setting {config_attr}: model({model_id}) -> Tokenizer({tokenizer_id})")
                    setattr(model.config, config_attr, tokenizer_id)
            else:
                print(f"YL_warning:  {token_type} not exist in tokenizer")

        # 3. Special handling for pad_token (ensure consistency)
        if model.config.pad_token_id is None or tokenizer.pad_token_id is None:
            if tokenizer.pad_token is None:
                tokenizer.add_special_tokens({'pad_token': '[PAD]'})
            model.config.pad_token_id = tokenizer.pad_token_id
            print(f"Setting pad_token_id: {tokenizer.pad_token_id}")
        elif model.config.pad_token_id != tokenizer.pad_token_id:
            print(
                f"YL_warning: Setting model.config.pad_token_id ({model.config.pad_token_id}) -> tokenizer.pad_token_id ({tokenizer.pad_token_id})")
            model.config.pad_token_id = tokenizer.pad_token_id

        # 4. Synchronize vocabulary size
        if model.config.vocab_size != len(tokenizer):
            print(f"\nYL_warning: model's Vocabulary Size ({model.config.vocab_size}) != tokenizer's Vocabulary Size ({len(tokenizer)})")
            print(f"Adjust the size of token embeddings: {model.config.vocab_size} -> {len(tokenizer)}")
            model.resize_token_embeddings(len(tokenizer))

        # 5. Print status after synchronization
        print("\n=== After synchronization  ===")
        print("\n--- Special tokens of Model ---")
        for attr in dir(model.config):
            if attr.endswith('_token_id') and not attr.startswith('__'):
                print(f"{attr}: {getattr(model.config, attr)}")

        print("\n--- Special tokens of Tokenizer ---")
        for token_type in special_tokens:
            token = getattr(tokenizer, token_type, None)
            token_id = getattr(tokenizer, f"{token_type}_id", None)
            print(f"{token_type}: '{token}' (ID: {token_id})")

    sync_special_tokens(model, tokenizer)

    # Define data encoding function
    def encode(examples):
        return tokenizer(examples["text"],truncation=True, padding='max_length',max_length=training_args.model_max_length)

    # Load dataset
    if data_args.split > 0:
        # If validation set is needed, split the dataset
        data_sets = load_dataset('text', data_files=data_args.train_data, split='train')
        data_sets = data_sets.train_test_split(test_size=data_args.split, seed=training_args.seed,shuffle=data_args.shuffle)
        data_sets = data_sets.map(encode, batched=True)
        print("Data sets split into train and test:", data_sets)
        for split, dataset in data_sets.items():
            print(f"{split} dataset size: {len(dataset)}")
        trainset, validset = data_sets['train'], data_sets['test']

    else:
        # No validation set needed, only load training data
        trainset = load_dataset('text', data_files=data_args.train_data, split='train')
        print("Dataset loaded (no validation split):", trainset)
        trainset = trainset.map(encode, batched=True)
        validset = None
        print("Training dataset size:", len(trainset))

    # Initialize data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=model_args.is_mlm)
    print(f"Data collator initialized: {data_collator}")

    # Call training function
    train(training_args, model, trainset, validset, data_collator, output_dir,
          resume_from_checkpoint=model_args.load_checkpoint)
    print(f"Training completed: {train}")

    # Save tokenizer
    tokenizer.save_pretrained(training_args.output_dir)


if __name__ == '__main__':
    main()
    print("Pretrain okokokok")




