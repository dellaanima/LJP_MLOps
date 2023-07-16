import os
import subprocess
import sys
import argparse
import logging
import numpy as np

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    
def parser_args(train_notebook=False):
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--model_id", default='lawcompany/KLAID_LJP_base')
    parser.add_argument("--tokenizer_id", default='lawcompany/KLAID_LJP_base')
    parser.add_argument("--dataset_name", type=str, default='lawcompany/KLAID')
    parser.add_argument("--small_subset_for_debug", type=bool, default=True)
    parser.add_argument("--train_dir", type=str, default='/opt/ml/processing/train')
    parser.add_argument("--validation_dir", type=str, default='/opt/ml/processing/validation')    
    parser.add_argument("--test_dir", type=str, default='/opt/ml/processing/test')
    parser.add_argument("--transformers_version", type=str, default='4.17.0')
    parser.add_argument("--pytorch_version", type=str, default='1.10.2')    

        
    if train_notebook:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()
    return args
    
    
if __name__ == "__main__":
    args = parser_args()
    
    install(f"torch=={args.pytorch_version}")
    install(f"transformers=={args.transformers_version}")
    install("datasets==1.18.4")
    
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['fact'], padding='max_length', max_length=512, truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset(args.dataset_name, split=['train[:1%]', 'train[-1%:]'])
    
    if args.small_subset_for_debug:
        train_dataset = train_dataset.shuffle().select(range(1000))
        test_dataset = test_dataset.shuffle().select(range(1000))

    # tokenize dataset
    train_dataset = train_dataset.map(tokenize, batched=True)
    test_dataset = test_dataset.map(tokenize, batched=True)

    # set format for pytorch
    train_dataset =  train_dataset.rename_column("laws_service_id", "labels")
    train_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    test_dataset = test_dataset.rename_column("laws_service_id", "labels")
    test_dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    
    train_dataset.save_to_disk(args.train_dir)
    test_dataset.save_to_disk(args.test_dir)