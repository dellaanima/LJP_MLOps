#!/bin/bash
# This script installs necessary packages for your SageMaker Studio environment

set -eux

# PARAMETERS
PACKAGES=(
    "argparse"
    "torchvision==0.14.1"
    "awscli==1.27.68"
    "boto3==1.26.68"
    "botocore==1.29.68"
    "datasets==1.18.4"
    "sagemaker==2.143.0"
    "s3fs==0.4.2"
    "s3transfer==0.6.0"
    "transformers==4.17.0"
    "nvidia-cublas-cu11==11.10.3.66"
    "nvidia-cuda-nvrtc-cu11==11.7.99"
    "nvidia-cuda-runtime-cu11==11.7.99"
    "nvidia-cudnn-cu11==8.5.0.96"
)

# Install each package
for package in "${PACKAGES[@]}"; do
    pip install --upgrade "$package"
done

