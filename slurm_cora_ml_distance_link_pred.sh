#!/bin/bash

#SBATCH --qos bbgpu
#SBATCH --gres gpu:k20:1
#SBATCH --job-name=coraMLDistanceLinkPred
#SBATCH --output=coraMLDistanceLinkPred_%A_%a.out
#SBATCH --error=coraMLDistanceLinkPred_%A_%a.err
#SBATCH --array=0-59
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8gb

PROJECT_DIR=/rds/projects/2018/hesz01/attributed_hyperbolic
ARR=(-r={5,3}" "-t={1,3}" "--dim={2,3,5,32,128}" --evaluate-link-prediction "{--no-attributes,--multiply-attributes,--jump-prob=0.05})

module purge; module load bluebear
module load apps/python3/3.5.2
module load apps/tensorflow/1.3.1-python-3.5.2-cuda-8.0.44
module load apps/keras/2.0.8-python-3.5.2-cuda-8.0.44

echo starting, ${ARR[${SLURM_ARRAY_TASK_ID}]}
python embedding/hyperbolic_embedding.py --dataset cora_ml --data-directory ${PROJECT_DIR}/data --no-load ${ARR[${SLURM_ARRAY_TASK_ID}]} 