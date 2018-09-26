#!/bin/bash

#SBATCH --job-name=noGPUCoraMLDistanceLinkPred
#SBATCH --output=noGPUCoraMLDistanceLinkPred_%A_%a.out
#SBATCH --error=noGPUCoraMLDistanceLinkPred_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G


PROJECT_DIR=/rds/projects/2018/hesz01/attributed_hyperbolic
# ARR=(--dim={3,5,10,32}" "--seed={0..29}" "--softmax" "--evaluate-link-prediction" "--{no-attributes,jump-prob={0.05,0.1}})
ARR=(--dataset={cora_ml,GrQc}" "--seed={0..29}" "--softmax" "--dim=5" --evaluate-link-prediction "--no-attributes)


module purge; module load bluebear
module load apps/cuda/8.0.44
module load apps/cudnn/6.0
-I${CUDNN_ROOT}/include/cudnn.h -L${CUDNN_ROOT}/lib64/libcudnn.so
module load apps/python3/3.5.2
module load apps/tensorflow/1.3.1-python-3.5.2
module load apps/keras/2.0.8-python-3.5.2

echo "staring dataset=cora_ml ct1 "${ARR[${SLURM_ARRAY_TASK_ID}]}
python embedding/hyperbolic_embedding.py \
--data-directory ${PROJECT_DIR}/data --patience 1000 --lr .3 -b 10 -e 1000 --no-load ${ARR[${SLURM_ARRAY_TASK_ID}]} \
--context-size 1 
echo "completed dataset=cora_ml ct1"${ARR[${SLURM_ARRAY_TASK_ID}]}
