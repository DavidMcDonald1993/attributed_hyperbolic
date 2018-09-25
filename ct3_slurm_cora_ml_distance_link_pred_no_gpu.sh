#!/bin/bash

#SBATCH --job-name=ct3noGPUCoraMLDistanceLinkPred
#SBATCH --output=ct3noGPUCoraMLDistanceLinkPred_%A_%a.out
#SBATCH --error=ct3noGPUCoraMLDistanceLinkPred_%A_%a.err
#SBATCH --array=0-29
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G



PROJECT_DIR=/rds/projects/2018/hesz01/attributed_hyperbolic
# ARR=({--softmax,-r={5,3}" "-t={1,3}}" "--dim={2,3,5,32,128}" --evaluate-link-prediction "{--no-attributes,--multiply-attributes,--jump-prob=0.05})
ARR=(--seed={0..29}" "--softmax" "--dim=3" --evaluate-link-prediction "--no-attributes)

module purge; module load bluebear
module load apps/cuda/8.0.44
module load apps/cudnn/6.0
-I${CUDNN_ROOT}/include/cudnn.h -L${CUDNN_ROOT}/lib64/libcudnn.so
module load apps/python3/3.5.2
module load apps/tensorflow/1.3.1-python-3.5.2
module load apps/keras/2.0.8-python-3.5.2

echo "staring dataset=cora_ml ct3 "${ARR[${SLURM_ARRAY_TASK_ID}]}
python embedding/hyperbolic_embedding.py --dataset cora_ml \
--data-directory ${PROJECT_DIR}/data --patience 1000 --lr .1 -e 1000 --no-load ${ARR[${SLURM_ARRAY_TASK_ID}]} \
--context-size 3 --model models_ct3 --plot plots_ct3 --logs logs_ct3 --test-results test_results_ct3
echo "completed dataset=cora_ml ct3 "${ARR[${SLURM_ARRAY_TASK_ID}]}