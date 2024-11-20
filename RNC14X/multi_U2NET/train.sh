#!/bin/bash
module load cudnn/8.2.1_cuda11.x
module load cuda/11.1
module load nccl/2.11.4-1_cuda11.1
export PYTHONUNBUFFERED=1
python -u u2net_train.py
