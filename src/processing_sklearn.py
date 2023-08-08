import os
import subprocess
import sys
import argparse
import logging
import numpy as np
import boto3
import datetime
from datetime import datetime 

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
    transformers_version = "4.17.0" 
    install(f"transformers=={transformers_version}")
    install("datasets==1.18.4")
    
    ## Data Crawling 

    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    # download tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    # tokenizer helper function
    def tokenize(batch):
        return tokenizer(batch['fact'], padding='max_length', max_length=512, truncation=True)

    # load dataset
    train_dataset, test_dataset = load_dataset(args.dataset_name, split=['train[:80%]', 'train[80%:]'])

    # train dataset to dataframe 
    import pandas as pd
    train_df = pd.DataFrame(train_dataset)

    # local file Path
    local_file_path = "/opt/ml/processing/collected_data.csv"  

    # Download collected Data from S3 bucket
    s3 = boto3.client("s3")
    s3_bucket = "sagemaker-us-east-1-353411055907"
    current_date_str = f"data_{datetime.now().strftime('%Y-%m-%d %H')}"
    file_name = f'GP-LJP-mlops/data/collected_data/{current_date_str}.csv'
    s3.download_file(s3_bucket, file_name, local_file_path)

    # Concatenate the original data + collected data 
    added_df = pd.read_csv(local_file_path, encoding='utf-8')
    merged_df = pd.concat([train_df, added_df], axis=0)

    # Convert the merged DataFrame back to the Hugging Face Dataset class format
    from datasets import Dataset
    train_dataset = Dataset.from_pandas(merged_df)

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
