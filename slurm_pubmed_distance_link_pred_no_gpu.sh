#!/bin/bash

#SBATCH --job-name=noGPUPubmedDistanceLinkPred
#SBATCH --output=noGPUPubmedDistanceLinkPred_%A_%a.out
#SBATCH --error=noGPUPubmedDistanceLinkPred_%A_%a.err
#SBATCH --array=0-59
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=32G



PROJECT_DIR=/rds/projects/2018/hesz01/attributed_hyperbolic
# ARR=({--softmax,-r={5,3}" "-t={1,3}}" "--dim={2,3,5,32,128}" --evaluate-link-prediction "{--no-attributes,--multiply-attributes,--jump-prob=0.05})
ARR=(--dim=3" "--seed={0..29}" "--softmax" "--evaluate-link-prediction" "--{no-attributes,jump-prob=0.05})

module purge; module load bluebear
module load apps/cuda/8.0.44
module load apps/cudnn/6.0
-I${CUDNN_ROOT}/include/cudnn.h -L${CUDNN_ROOT}/lib64/libcudnn.so
module load apps/python3/3.5.2
module load apps/tensorflow/1.3.1-python-3.5.2
module load apps/keras/2.0.8-python-3.5.2

echo "staring dataset=pubmed "${ARR[${SLURM_ARRAY_TASK_ID}]}
python embedding/hyperbolic_embedding.py --dataset pubmed --data-directory ${PROJECT_DIR}/data --no-load ${ARR[${SLURM_ARRAY_TASK_ID}]} 
echo "completed dataset=pubmed "${ARR[${SLURM_ARRAY_TASK_ID}]}
