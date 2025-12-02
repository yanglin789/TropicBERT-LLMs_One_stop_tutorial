"""Unified Genomic Sequence Classification/Regression Framework

This code is a general-purpose deep learning framework based on HuggingFace Transformers,
specifically designed for downstream tasks related to genomic sequences
(such as promoter prediction, enhancer identification, epigenetic marker prediction, etc.).

The framework supports three main task types: regression, binary classification, and multi-classification,
and provides a complete workflow for data processing, model training, evaluation, and result analysis.
============================================================================================
Usage Instructions:
    1. Prepare a dataset in CSV format (including a sequence column and a label column)
    2. Configure ModelArguments, DataArguments, and TrainingArguments
    3. Run the script to start training and evaluation

Application Scenarios:
    - Genomic functional element prediction (promoters, enhancers, silencers, etc.)
    - Epigenetic marker prediction (DNA methylation, histone modifications, etc.)
    - Protein-DNA interaction prediction
    - Pathogen sequence classification
    - Any other bioinformatics tasks requiring sequence classification/regression

Core Features and Advantages:

Multi-task Versatility:
    - Supports regression, binary classification, and multi-classification tasks, suitable for various genomic prediction tasks
    - Automatically handles label mapping and class imbalance issues
    - Built-in multiple evaluation metrics (e.g., MSE, F1, AUC, MCC, etc.)

Pedagogically Friendly Design:
    - Modular code structure for easy understanding and modification
    - Detailed parameter configurations and comments, suitable for teaching purposes
    - Includes end-to-end examples covering data preprocessing, model construction, training, and evaluation

Advanced Features:
    - Optional reinitialization of the model classification layer
    - Flexible tokenizer synchronization mechanism
    - Comprehensive result saving and analysis capabilities

Genomic Sequence-Specific Processing:
    - Supports k-mer splitting (dividing sequences into substrings of length k)
    - Automatically handles genomic sequences of varying lengths
    - Supports custom sequence column names and label column names

Main References:
    - huggingface transformers(https://github.com/huggingface/transformers)
    - DNABERT (https://github.com/jerryji1993/DNABERT)
    - nucleotide-transformer(https://github.com/instadeepai/nucleotide-transformer)
    - Plant_DNA_LLMs (https://github.com/zhangtaolab/Plant_DNA_LLMs)

Example Command:
    python script.py \
        --model_name_or_path dna_bert \
        --train_data_path train.csv \
        --eval_data_path dev.csv \
        --test_data_path test.csv \
        --train_task classification \
        --output_dir ./results \

Code Information:
    Author: YangLin
    Team: Xia_lab
    Date: November 20, 2025
"""
import os
import random
import torch
import transformers
import csv
import numpy as np
# from transformers import AutoConfig,AutoModelForTokenClassification
from transformers import TrainingArguments, Trainer,HfArgumentParser
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from typing import Optional
from datasets import Dataset, DatasetDict, load_dataset
from dataclasses import dataclass, field
from scipy.stats import spearmanr, pearsonr
from scipy.special import softmax
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, matthews_corrcoef
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score,roc_auc_score,top_k_accuracy_score


########################################################################################################################
## 1. Parameter Definition ##
########################################################################################################################

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default=None)    #path to the pre-trained model or model identifier in the Hugging Face model library
    tokenizer_path: Optional[str] = field(default=None)        #name or path of the pre-trained tokenizer
    train_task: Optional[str] = field(default='classification')  # regression (regression), classification (binary), multi-classification (multi-class)
    load_checkpoint: Optional[str] = field(default=None)
    reinit_classifier_layer: Optional[bool] = field(default=False)  # Initialize model classification layer
    sync_tokenizer_to_model: Optional[bool] = field(default=False) # Whether to additionally synchronize tokenizer to model, added by YL, mainly used after modifying tokenizer


@dataclass
class DataArguments:
    # labels: Optional[str] = field(default=None)  #Labels 'Not promoter;Core promoter'
    train_data_path: Optional[str] = field(default=None) #training data path
    eval_data_path: Optional[str] = field(default=None) #validation data path
    test_data_path: Optional[str] = field(default=None) #test data path
    split_train_data_only: float = field(default=0.1)   #When only training data is available, the proportion of validation data to split
    shuffle_train: bool = field(default=False)  #whether to randomly shuffle training data
    samples: Optional[int] = field(default=1e10)
    seq_column_name: str = field(default='sequence')
    label_column_name: str = field(default='label')
    seq_kmer_splice: Optional[int] = field(default=None)# In data processing, whether to truncate the sequence by k and add spaces (if a value is passed, split according to the value)


@dataclass
class TrainingArguments(TrainingArguments):
    seed: int = field(default=42) #random seed
    model_max_length: int = field(default=512) #maximum sequence length of the model
    tokenizer_use_fast: bool = field(default=True)
    output_dir: str = field(default="05_output")   #output directory
    run_name: str = field(default="runs")  #run name
    optim: str = field(default="adamw_torch")   #optimizer type
    gradient_accumulation_steps: int = field(default=2) #gradient accumulation steps
    per_device_train_batch_size: int = field(default=8) #training batch size per device
    per_device_eval_batch_size: int = field(default=8)  #evaluation batch size per device
    num_train_epochs: int = field(default=5)    #total number of training epochs
    fp16: bool = field(default=False)   #whether to use half-precision floating point (FP16)
    bf16: bool = field(default=False)   #whether to use Bfloat16
    logging_steps: Optional[int] = field(default=100)   #logging steps
    logging_strategy: str = field(default='steps')   #logging strategy
    save_steps: Optional[int] = field(default=1000)  #steps to save model
    save_strategy: str = field(default='steps') #save strategy
    eval_steps: Optional[int] = field(default=100) #evaluation steps
    eval_strategy: str = field(default='steps') #evaluation strategy
    warmup_ratio: float = field(default=0.05)   #warmup steps ratio for learning rate
    weight_decay: float = field(default=0.01)   #weight decay
    learning_rate: float = field(default=5e-5)  #learning rate
    save_total_limit: int = field(default=5)    #maximum number of saved models
    load_best_model_at_end: bool = field(default=True)  #load best model at the end of training
    metric_for_best_model: Optional[str] = field(default=None) #'f1'
    greater_is_better: Optional[bool] = field(default=None) # Whether the evaluation metric is better when larger
    cache_dir: Optional[str] = field(default=None) # Specify cache directory for network downloads, if not set, default cache directory is ~/.cache/huggingface/transformers


########################################################################################################################
## 2. Function Definition ##
########################################################################################################################
# Function: Set random seed to ensure reproducibility of experiments
def setup_seed(seed):
    """Set random seeds for Python, NumPy, PyTorch, and Transformers to ensure experimental reproducibility"""
    random.seed(seed)  #set random seed for Python's built-in random library
    np.random.seed(seed)  #set random seed for NumPy library
    torch.manual_seed(seed)  #set CPU random seed for PyTorch
    torch.cuda.manual_seed_all(seed)  #ensure all GPU operations have the same seed
    transformers.set_seed(seed)  #set global random seed for transformers


def load_and_split_csv_dataset(data_args, training_args):
    """
    Load CSV format dataset and split and shuffle according to parameters
    This function loads the dataset based on the provided CSV file path and splits out a validation set by proportion when only training data is available
    Parameters:
        data_args: Object containing data path and other parameters
        training_args: Object containing training parameters, used to set random seed
    Returns:
        DatasetDict: Object containing train, dev, test three dataset splits
    """
    print("#"*100,"\nUsing csv file")
    data_files_path = {}

    # Check if input file is in CSV format
    if data_args.train_data_path.endswith(".csv"):
        # Collect all data file paths
        if data_args.train_data_path:
            data_files_path['train'] = data_args.train_data_path
        if data_args.eval_data_path:
            data_files_path['dev'] = data_args.eval_data_path
        if data_args.test_data_path:
            data_files_path['test'] = data_args.test_data_path

        # Load dataset based on data file paths
        if 'dev' in data_files_path or 'test' in data_files_path:
            # When validation or test data is available, load directly
            dataset = load_dataset('csv', data_files=data_files_path)
        else:
            # When only training data is available, load and split
            dataset = load_dataset('csv', data_files=data_args.train_data_path)
            dataset = dataset['train'].train_test_split(
                test_size=data_args.split_train_data_only,
                seed=training_args.seed,
                shuffle=False  #do not shuffle dataset during split (this parameter defaults to True)
            )

        # If data shuffling is needed, randomly shuffle the training set
        if data_args.shuffle_train:
            dataset["train"] = dataset["train"].shuffle(training_args.seed)

    else:
        raise ValueError("No valid data source provided.")

    return dataset


def get_num_labels_and_mappings(model_args_i, data_args_i, dataset_i):
    def f_get_label_list(data_args_f,raw_datasets: DatasetDict) -> list[str]:
        all_labels_i = []
        label_column_name = data_args_f.label_column_name
        for split in ["train", "dev", "test"]:
            if split in raw_datasets:
                all_labels_i.extend(raw_datasets[split][label_column_name])
        label_list_i = sorted(list(set(all_labels_i)))
        return label_list_i

    label_list = []
    if model_args_i.train_task == "regression":  # Numerical regression
        num_labels = 1
        label2id = None
        id2label = None

    elif model_args_i.train_task == "classification":  # Binary classification (yes/no)
        label_list = f_get_label_list(data_args_i, dataset_i)  # Get label list from training set

        if -1 in label_list:
            print("Label -1 found in label list, removing it.")
            label_list = [label for label in label_list if label != -1]

        label_list.sort()  # Sort label list
        num_labels = len(label_list)  # Calculate number of labels
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

        # Generate label mapping relationships
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for idx, label in enumerate(label_list)}

        # Print label mapping relationships and related information
        print("#" * 100)
        print(f"{model_args_i.train_task} - Label Mapping:")
        print(type(label_list), f"labels numbers: {len(label_list)}\n{label_list}")
        for k, v in label2id.items():
            print(f"Original Label {k} -> Model Input {v}")
        print("#" * 100)


    elif model_args_i.train_task == "multi-classification":  # Multi-classification (can only correspond to 1 type)
        label_list = f_get_label_list(data_args_i, dataset_i)  # Get label list from training set

        if -1 in label_list:
            print("Label -1 found in label list, removing it.")
            label_list = [label for label in label_list if label != -1]

        label_list.sort()  # Sort label list
        num_labels = len(label_list)  # Calculate number of labels
        if num_labels <= 1:
            raise ValueError("You need more than one label to do classification.")

        # Generate label mapping relationships
        label2id = {label: idx for idx, label in enumerate(label_list)}
        id2label = {idx: label for idx, label in enumerate(label_list)}

        # Print label mapping relationships and related information
        print("#" * 100)
        print(f"{model_args_i.train_task} - Label Mapping:")
        print(type(label_list), f"labels numbers: {len(label_list)}\n{label_list}")
        for k, v in label2id.items():
            print(f"Original Label {k} -> Model Input {v}")
        print("#" * 100)

    else:
        num_labels = 0
        label2id = None
        id2label = None
        print("YL ERRO:num_labels=0")


    return num_labels, label2id, id2label


def preprocess_seq_data(examples, data_args, tokenizer, training_args):
    if data_args.seq_kmer_splice is not None and data_args.seq_kmer_splice > 0:
        # Use adding spaces to k-mer sequences after tokenization
        def f_seq2kmer(seqs, k):
            all_kmers = []
            for seq in seqs:
                kmer = [seq[x:x + k].upper() for x in range(len(seq) + 1 - k)]
                kmers = " ".join(kmer)
                all_kmers.append(kmers)
            return all_kmers
        sequence = f_seq2kmer(examples['sequence'], data_args.seq_kmer_splice)  # Call seq2kmer to split sequence into k-mer substrings
        # Use tokenizer to tokenize k-mer sequences, and pad/truncate to fixed length model_max_length
        tokenized_seq = tokenizer(
            sequence,
            truncation=True,
            padding='max_length',
            max_length=training_args.model_max_length)
        return tokenized_seq
    else:
        # Directly tokenize
        seq_column_name = data_args.seq_column_name
        print(f"seq_column_name:{seq_column_name}")
        sequence_seq = examples[seq_column_name]  # Filtered characters
        tokenized_seq = tokenizer(
            sequence_seq,
            truncation=True,
            padding='max_length',
            max_length=training_args.model_max_length)
        return tokenized_seq


def print_statistics(split, dataset):
    lengths = [len(sample['input_ids']) for sample in dataset]
    print(f"{split} dataset statistics:")
    print(f"  dataset type:{type(dataset)}: \n {dataset}")
    print(f"  ##  Max length: {max(lengths)}","##",f"  Min length: {min(lengths)}","##",f"  Median length: {np.median(lengths)}")
    print(f"  Example input_ids (first 3 samples):")
    for i in range(min(3, len(dataset))):  # Print up to 3 examples
        print(f"    Sample {i + 1} 'input_ids': {dataset[i]['input_ids']}")


def load_model(model_args, num_labels, id2label, label2id, training_args):
    # 7-1-Load model
    if model_args.train_task == "regression":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,  # Specify model name or path
            # config=config,  # Model configuration
            cache_dir=training_args.cache_dir,  # Specify cache directory
            num_labels=num_labels,  # Set to 1 for regression task
            trust_remote_code=True,  # Trust remote code, allow loading models with custom code
            problem_type="regression",
            # ignore_mismatched_sizes=True      # Optional parameter, ignore model size mismatch issues
        )
    elif model_args.train_task == "classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,  # Specify model name or path
            # config=config,  # Model configuration
            cache_dir=training_args.cache_dir,  # Specify cache directory
            num_labels=num_labels,  # Set number of labels
            trust_remote_code=True,  # Trust remote code, allow loading models with custom code
            # ignore_mismatched_sizes=True      # Optional parameter, ignore model size mismatch issues
        )
    elif model_args.train_task == "multi-classification":
        model = AutoModelForSequenceClassification.from_pretrained(
            model_args.model_name_or_path,  # Specify model name or path
            # config=config,  # Model configuration
            cache_dir=training_args.cache_dir,  # Specify cache directory for network downloads
            num_labels=num_labels,  # Set number of labels
            id2label=id2label,  # Token ID to label name mapping
            label2id=label2id,  # Label name to Token ID mapping
            problem_type="single_label_classification",
            trust_remote_code=True,  # Trust remote code, allow loading models with custom code
            # ignore_mismatched_sizes=True  # Optional parameter, ignore model size mismatch issues
        )
    else:
        raise ValueError(f"Unknown model_args.train_task: {model_args.train_task}")  # Raise exception or return default value

    return model


def init_model_weights(model, model_args):
    print("#" * 100, f"\nModel:\n{model}")
    if model_args.reinit_classifier_layer:
        print("\nInitializing classification layers now....")
        if hasattr(model, "score"):
            old_score_weight = model.score.weight.data.clone()
            torch.nn.init.xavier_uniform_(model.score.weight)
            new_score_weight = model.score.weight.data
            print(f"Score layer - Before: mean={old_score_weight.mean():.6f}, std={old_score_weight.std():.6f}| After: mean={new_score_weight.mean():.6f}, std={new_score_weight.std():.6f}")
        if hasattr(model, 'classifier'):
            old_classifier_weight = model.classifier.weight.data.clone()
            torch.nn.init.xavier_uniform_(model.classifier.weight)
            new_classifier_weight = model.classifier.weight.data
            print(f"Classifier layer - Before: mean={old_classifier_weight.mean():.6f}, std={old_classifier_weight.std():.6f} | After:  mean={new_classifier_weight.mean():.6f}, std={new_classifier_weight.std():.6f}")

    return model


# Select and return the corresponding evaluation metrics function based on task type
def evaluate_metrics(task_name):
    if task_name == "regression":    #numerical regression
        f2_compute_metrics = regression_metrics()
    elif task_name == "classification": #binary classification (yes/no)
        f2_compute_metrics = classification_metrics()
    elif task_name == "multi-classification":   #multi-classification (can only correspond to 1 type)
        f2_compute_metrics = multi_classifications_metrics()
    else:
        raise ValueError(f"Unknown task_name: {task_name}")  # Raise exception or return default value
    return f2_compute_metrics

def regression_metrics():
    """ Function to calculate metrics for evaluating regression model performance"""
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits.squeeze()
        labels = labels.squeeze()

        assert len(logits) == len(labels), "logits and labels have inconsistent lengths"

        # Calculate various regression metrics
        mse = mean_squared_error(labels, logits)  # Mean Squared Error (MSE): Average of squared differences between predicted and actual values
        mae = mean_absolute_error(labels, logits)  # Mean Absolute Error (MAE): Average of absolute differences between predicted and actual values
        r2 = r2_score(labels, logits)  # R-squared (R2): Proportion of variance explained by the model, range (-∞,1]
        spearman_corr, _ = spearmanr(logits, labels)  # Spearman rank correlation coefficient: Measures monotonic relationship between predicted and actual values
        pearson_corr, _ = pearsonr(logits, labels)  # Pearson correlation coefficient: Measures linear correlation between predicted and actual values

        return {
            'mse': mse,  # Mean squared error
            'mae': mae,  # Mean absolute error
            'r2': r2,  # R-squared value
            'spearmanr': spearman_corr,  # Spearman correlation coefficient
            'pearsonr': pearson_corr,  # Pearson correlation coefficient
        }

    return compute_metrics

def classification_metrics():
    """Function to calculate metrics for evaluating binary classification model performance"""
    def f_compute_metrics_sklearn(eval_pred):
        logits, labels = eval_pred
        # Ensure predictions needed for classification tasks can be correctly extracted
        if isinstance(logits, tuple):
            logits = logits[0]
        # Take the index of the maximum value in the last dimension, i.e., the predicted class label
        predictions = np.argmax(logits, axis=-1) #np.argmax returns the index of the maximum value in the last dimension (corresponding to the predicted class label)

        # Calculate various metrics
        accuracy = accuracy_score(labels, predictions)  # Accuracy: Proportion of correctly predicted samples to total samples
        f1 = f1_score(labels, predictions, average='binary')  # F1 score: Harmonic mean of precision and recall
        precision = precision_score(labels, predictions, average='binary')  # Precision: Proportion of actual positive samples among those predicted as positive
        recall = recall_score(labels, predictions, average='binary')  # Recall: Proportion of positive samples predicted as positive among actual positive samples
        matthews_correlation = matthews_corrcoef(labels, predictions)  # Matthews correlation coefficient: Metric for measuring classifier performance
        return {
            'accuracy': accuracy,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'matthews_correlation': matthews_correlation
        }
    return f_compute_metrics_sklearn

def multi_classifications_metrics():
    """Function to calculate metrics for evaluating single-label multi-classification model performance"""
    def f_compute_metrics(eval_pred):
        logits, labels = eval_pred
        logits = logits[0] if isinstance(logits, tuple) else logits

        # Calculate prediction probabilities and predicted classes
        pred_probs = softmax(logits, axis=1)
        predictions = np.argmax(pred_probs, axis=1)
        labels = np.array(labels)
        print("##"*50,"---eval---")
        print("\n pred_probs",pred_probs,"\n predictions",predictions,"\n labels",labels)

        # Calculate various metrics
        metrics = {
            "accuracy": accuracy_score(labels, predictions),
            "precision_micro": precision_score(labels, predictions, average="micro"),
            "precision_macro": precision_score(labels, predictions, average="macro"),
            "precision_weighted": precision_score(labels, predictions, average="weighted"),
            "recall_micro": recall_score(labels, predictions, average="micro"),
            "recall_macro": recall_score(labels, predictions, average="macro"),
            "recall_weighted": recall_score(labels, predictions, average="weighted"),
            "f1_micro": f1_score(labels, predictions, average="micro"),
            "f1_macro": f1_score(labels, predictions, average="macro"),
            "f1_weighted": f1_score(labels, predictions, average="weighted"),
            "mcc": matthews_corrcoef(labels, predictions),
            "roc_auc_ovr": roc_auc_score(labels, pred_probs, multi_class="ovr"),
            "roc_auc_ovo": roc_auc_score(labels, pred_probs, multi_class="ovo"),
            "top_1_accuracy": top_k_accuracy_score(labels, pred_probs, k=1),
            "top_2_accuracy": top_k_accuracy_score(labels, pred_probs, k=2),
            "top_3_accuracy": top_k_accuracy_score(labels, pred_probs, k=3),
            "top_5_accuracy": top_k_accuracy_score(labels, pred_probs, k=5),
        }
        return metrics

    return f_compute_metrics


# Save prediction results for each data during evaluation
def save_test_results(model_args, predictions, labels, input_sequences, original_labels, id2label, training_args):
    if model_args.train_task == "regression": # Regression
        regression_results(predictions, labels, input_sequences, original_labels, training_args)

    elif model_args.train_task == "classification":  # Binary classification
        classification_results(predictions, labels, input_sequences, original_labels, id2label, training_args)

    elif model_args.train_task == "multi-classification":  # Multi-classification
        multi_classification_results(predictions, labels, input_sequences, original_labels, id2label, training_args)

    else:
        print(f"Unknown task type: {model_args.train_task}")

def regression_results(predictions, labels, input_sequences, original_labels, training_args):
    """Save prediction results for regression task"""
    # Process predictions (extract first element from 2D array)
    if isinstance(predictions, np.ndarray):
        predictions = predictions.flatten()  # If it's a 1D array, flatten directly
    elif isinstance(predictions, list) and all(isinstance(x, list) and len(x) == 1 for x in predictions):
        predictions = [x[0] for x in predictions]  # If it's [[-0.011], [-0.067], ...], extract first element

    # Ensure labels are iterable list/array
    if isinstance(labels, np.ndarray):
        labels = labels.tolist()
    elif not isinstance(labels, list):
        labels = list(labels)

    # Ensure input_sequences and original_labels are lists
    if not isinstance(input_sequences, list):
        input_sequences = list(input_sequences)
    if not isinstance(original_labels, list):
        original_labels = list(original_labels)

    # Prepare CSV data
    csv_data = []
    for seq, orig_label, pred_label, label in zip(input_sequences, original_labels, predictions, labels):
        csv_data.append({
            'predicted_label': float(pred_label),  # Ensure it's a float
            'label': float(label),  # True label (continuous value)
            'original_label': float(orig_label) if orig_label is not None else None,  # Original label
            'sequence': seq  # Input sequence
        })

    # Write to CSV file
    output_predictions_path = os.path.join(training_args.output_dir, "test_predictions_results.csv")
    with open(output_predictions_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['predicted_label', 'label', 'original_label', 'sequence'])
        writer.writeheader()
        writer.writerows(csv_data)

    print("YL: The test results of the Regression task have been saved.")

def classification_results(predictions, labels, input_sequences, original_labels, id2label, training_args):
    """Save prediction results for binary classification task"""
    # Process prediction results - binary classification case (directly take probability values)
    if len(predictions.shape) > 1 and predictions.shape[1] == 1:
        # If output dimension is 1, this could be regression or binary classification
        predicted_probs = predictions.flatten()
        predicted_classes = (predicted_probs > 0.0).astype(int)  # Use 0 as threshold, if probability then use 0.5
    elif len(predictions.shape) > 1 and predictions.shape[1] == 2:
        # Binary classification, use softmax or sigmoid
        # Normalize original logits with softmax
        predictions_normalized = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        predicted_classes = np.argmax(predictions_normalized, axis=1)  # Get predicted class index
        predicted_probs = predictions_normalized[:, 1]  # Take probability of positive class (index 1)
    else:
        # If it's a 1D array, apply sigmoid
        predicted_probs = 1 / (1 + np.exp(-predictions.flatten()))
        predicted_classes = (predicted_probs > 0.5).astype(int)

    # Ensure all data is in list format
    input_sequences = list(input_sequences) if not isinstance(input_sequences, list) else input_sequences
    original_labels = list(original_labels) if not isinstance(original_labels, list) else original_labels
    labels = list(labels) if not isinstance(labels, list) else labels

    print("Sample predicted_classes first 3 example:", predicted_classes[:3])
    print("Sample predicted_probs first 3 example:", predicted_probs[:3])

    # Prepare CSV data
    csv_data = []
    for seq, orig_label, pred_class, pred_prob, true_label in zip(
            input_sequences, original_labels, predicted_classes, predicted_probs, labels):
        # Map index back to label names
        pred_class_name = id2label.get(pred_class, str(pred_class))  # Predicted class name
        true_label_name = id2label.get(true_label, str(true_label))  # True class name

        csv_data.append({
            'predicted_class_idx': int(pred_class),  # Predicted class index
            'predicted_class_name': pred_class_name,  # Predicted class name
            'predicted_probability': float(pred_prob),  # Predicted probability
            'true_label_idx': int(true_label),  # True class index
            'true_label_name': true_label_name,  # True class name
            'original_label': orig_label,  # Retain original format (could be string or other)
            'sequence': seq
        })

    # Write to CSV file
    output_predictions_path = os.path.join(training_args.output_dir, "test_predictions_results.csv")
    with open(output_predictions_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['predicted_class_idx', 'predicted_class_name',
                                               'predicted_probability', 'true_label_idx',
                                               'true_label_name', 'original_label', 'sequence'])
        writer.writeheader()
        writer.writerows(csv_data)

    print("YL: The test results of the Binary Classification task have been saved.")

def multi_classification_results(predictions, labels, input_sequences, original_labels, id2label, training_args):
    """Save prediction results for single-label multi-classification task"""
    # Process prediction results - first normalize with softmax, then take the class with maximum probability
    if len(predictions.shape) > 1 and predictions.shape[1] > 1:
        # Normalize original logits with softmax
        predictions_normalized = np.exp(predictions) / np.sum(np.exp(predictions), axis=1, keepdims=True)
        predicted_classes = np.argmax(predictions_normalized, axis=1)  # Get predicted class index
        predicted_probs = np.max(predictions_normalized, axis=1)  # Get maximum probability value after normalization

        # Get top-3 prediction results
        top3_indices = np.argsort(-predictions_normalized, axis=1)[:, :3]  # Take indices of top 3 probabilities
        top3_probs = np.take_along_axis(predictions_normalized, top3_indices, axis=1)  # Get corresponding probability values
    else:
        raise ValueError(f"The shape of the prediction results for the multi-classification task is incorrect:\n{predictions.shape}")

    # Ensure all data is in list format
    input_sequences = list(input_sequences) if not isinstance(input_sequences, list) else input_sequences
    original_labels = list(original_labels) if not isinstance(original_labels, list) else original_labels
    labels = list(labels) if not isinstance(labels, list) else labels
    print("predictions_normalized min/max:", predictions_normalized.min(), predictions_normalized.max())
    print("predictions_normalized shape:", predictions_normalized.shape)
    print("Sample predictions_normalized first 3 example:\n", predictions_normalized[:3])  # View normalized output of first 3 samples

    # Prepare CSV data
    csv_data = []
    for seq, orig_label, pred_class_idx, pred_prob, true_label_idx, top_indices, top_probs in zip(
            input_sequences, original_labels, predicted_classes, predicted_probs, labels, top3_indices, top3_probs):

        # Map index back to label names
        pred_class_name = id2label.get(pred_class_idx, str(pred_class_idx))  # Predicted class name
        true_label_name = id2label.get(true_label_idx, str(true_label_idx))  # True class name

        # Process top-3 labels and probabilities
        top3_labels = []
        top3_probs_list = []
        for i in range(3):
            if i < len(top_indices):  # Ensure index is valid, skip invalid values (third prediction in binary classification case)
                top3_labels.append(id2label.get(top_indices[i], str(top_indices[i])))
                top3_probs_list.append(float(top_probs[i]))
        # If fewer than 3 valid predictions, fill with empty strings and -1
        while len(top3_labels) < 3:
            top3_labels.append("")
            top3_probs_list.append(-1.0)

        csv_data.append({
            'predicted_class_idx': int(pred_class_idx),  # Predicted class index
            'predicted_class_name': pred_class_name,  # Predicted class name
            'predicted_probability': float(pred_prob),  # Predicted probability (already normalized)
            'true_label_idx': int(true_label_idx),  # True class index
            'true_label_name': true_label_name,  # True class name
            'original_label': orig_label,  # Retain original format (could be string or other)
            'sequence': seq,
            'top1_label': top3_labels[0],  # First prediction label
            'top1_prob': top3_probs_list[0],  # First prediction probability
            'top2_label': top3_labels[1],  # Second prediction label
            'top2_prob': top3_probs_list[1],  # Second prediction probability
            'top3_label': top3_labels[2],  # Third prediction label
            'top3_prob': top3_probs_list[2]  # Third prediction probability
        })

    # Write to CSV file
    output_predictions_path = os.path.join(training_args.output_dir, "test_predictions_results.csv")
    with open(output_predictions_path, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['predicted_class_idx', 'predicted_class_name',
                                               'predicted_probability', 'true_label_idx',
                                               'true_label_name', 'original_label', 'sequence',
                                               'top1_label', 'top1_prob',
                                               'top2_label', 'top2_prob',
                                               'top3_label', 'top3_prob'])
        writer.writeheader()
        writer.writerows(csv_data)

    print("YL: The test results of the multi-classification task have been saved.")


def f_sync_tokenizer_to_model(model, tokenizer):
    # Synchronize padding token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model.config.pad_token_id to: {tokenizer.pad_token_id}")
    elif tokenizer.pad_token_id != model.config.pad_token_id:
        print(f"Warning: Overriding model's pad_token_id ({model.config.pad_token_id}) with tokenizer's ({tokenizer.pad_token_id})")
        model.config.pad_token_id = tokenizer.pad_token_id
        print(f"Set model.config.pad_token_id to: {tokenizer.pad_token_id}")


    # Synchronize special tokens
    for token_type in ['bos_token_id', 'eos_token_id', 'unk_token_id', 'sep_token_id', 'cls_token_id']:
        if hasattr(model.config, token_type) and hasattr(tokenizer, token_type):
            tokenizer_val = getattr(tokenizer, token_type)
            model_val = getattr(model.config, token_type)
            if tokenizer_val is not None and model_val != tokenizer_val:
                print(f"Warning: {token_type} mismatch - model:{model_val} vs tokenizer:{tokenizer_val}")
                setattr(model.config, token_type, tokenizer_val)

    # Synchronize vocabulary size
    if model.config.vocab_size != len(tokenizer):
        print(f"Warning: Model vocab size ({model.config.vocab_size}) != tokenizer vocab size ({len(tokenizer)})")
        print(f'Resizing token embeddings: {model.config.vocab_size} -> {len(tokenizer)}')
        model.resize_token_embeddings(len(tokenizer))

########################################################################################################################
## 3. Training + Evaluation + Testing Functions ##
########################################################################################################################
def train():
    # 1-Parse command line hyperparameters
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    # 2-Set random seed
    setup_seed(training_args.seed)

    # 3-Load tokenizer
    if not model_args.tokenizer_path:
        model_args.tokenizer_path = model_args.model_name_or_path
    tokenizer = AutoTokenizer.from_pretrained(
        pretrained_model_name_or_path = model_args.tokenizer_path,  #tokenizer path
        cache_dir=training_args.cache_dir,  # network download storage path
        model_max_length = training_args.model_max_length,  #tokenizer maximum sequence length
        use_fast = training_args.tokenizer_use_fast,#use fast tokenizer
        trust_remote_code = True,
        # add_bos_token=False, # Whether to add start token--added by yl
        # add_eos_token=False, # Whether to add end token--added by yl
    )

    # 4-Load dataset
    dataset = load_and_split_csv_dataset(data_args, training_args)

    # 5-Determine training task type and number of labels
    num_labels, label2id, id2label = get_num_labels_and_mappings(model_args, data_args, dataset)

    # 6-Tokenize data
    # 6-1-Apply preprocessing function to dataset
    dataset_tokens = dataset.map(lambda examples: preprocess_seq_data(examples, data_args, tokenizer, training_args), batched=True)

    # 6-2-Calculate and output data related information
    for split_i, dataset_i in dataset_tokens.items():
        print_statistics(split_i, dataset_i)
    print("#" * 100)

    # 7-Load and modify model
    # 7-1-Load model
    model = load_model(model_args, num_labels, id2label, label2id, training_args)

    # 7-2-Synchronize model and tokenizer (manual synchronization---generally not needed YL)
    if model_args.sync_tokenizer_to_model:
        f_sync_tokenizer_to_model(model, tokenizer)

    # 8-Initialize model weights
    model = init_model_weights(model, model_args)

    # 9-Define evaluation metrics function
    compute_metrics = evaluate_metrics(model_args.train_task.lower())

    ######## Training ############################################################################

    # 10-Load Trainer
    # 10-1-Select Trainer class
    trainer_init = Trainer
    # 10-2-Initialize trainer
    print("##" * 50,"\ndataset_tokens:", dataset_tokens, "\n", "##" * 50)
    trainer = trainer_init(
        model=model,  # Model instance to be trained
        tokenizer=tokenizer,  # Associated tokenizer instance
        args=training_args,  # Training parameter configuration object
        train_dataset=dataset_tokens['train'],  # Training dataset (required)
        eval_dataset=dataset_tokens['dev'] ,    # Evaluation dataset
        compute_metrics=compute_metrics,  # Evaluation metrics calculation function
        # problem_type = "regression"  # or "multi_label_classification"
    )

    # 10-3-Train model
    if model_args.load_checkpoint:  # If checkpoint path is specified, load model from specified checkpoint and continue training
        trainer.train(model_args.load_checkpoint)   #trainer.train will load model state from this path.
    else:
        trainer.train()             # If no checkpoint is specified, train model from scratch

    # 11-1-Save model (Hugging Face/PyTorch format)
    trainer.save_model(training_args.output_dir)
    # 11-2-Directly save model and tokenizer
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    ######## Test Evaluation ############################################################################
    # 12-Test evaluation
    model.eval()  # If using trainer.predict, this line may not be needed
    results = trainer.predict(dataset_tokens['test'])

    # 12-1-Save model test evaluation metrics
    output_metrics_path = os.path.join(training_args.output_dir, "test_metrics.json")
    with open(output_metrics_path, "w") as outf:
        print(results.metrics, file=outf)

    # 12-2-Get prediction results
    predictions = results.predictions  # Model predictions
    labels = results.label_ids  # True labels

    input_sequences = dataset_tokens['test']['sequence']  # Input sequences
    original_labels = dataset_tokens['test']['label']  # Original labels (theoretically same as labels, but there may be subtle differences in practice)

    # 12-3-Get prediction results and execute corresponding test result output based on task type
    save_test_results(model_args, predictions, labels, input_sequences, original_labels, id2label, training_args)


if __name__ == "__main__":
    train()
    print("finetune okokokok")



