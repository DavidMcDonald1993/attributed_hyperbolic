#!/bin/bash

#SBATCH --job-name=noGPUexp
#SBATCH --output=noGPUexp_%A_%a.out
#SBATCH --error=noGPUexp_%A_%a.err
#SBATCH --array=0-2519
#SBATCH --time=1:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G
# SBATCH --mail-type ALL

DATA_DIR="/rds/homes/d/dxm237/data"

# ARR=(--dataset={cora_ml,citeseer,pubmed}" "--seed={0..29}" "--softmax" "--dim={5,10,25,50}" --evaluate-"{link,class}"-prediction "--{no-attributes,jump-prob={.05,.1,.2}})
ARR=(--dataset={cora_ml,citeseer,"ppi --only-lcc"}" "--seed={0..29}" "--softmax" "--dim={5,10,25,50}" --evaluate-link-prediction "--{no-attributes,jump-prob={.05,.1,.2,.5,.8,1.}})
# ARR=(--dataset=pubmed" "--seed={0..9}" "--softmax" "--dim={5,10,25,50}" --evaluate-"{link,class}"-prediction "--{no-attributes,jump-prob={.05,.1,.2,.5,.8,1.}})
# ARR=(--dataset=ppi" --only-lcc "--seed={0..29}" "--softmax" "--dim=5" --evaluate-"{link,class}"-prediction "--{no-attributes,jump-prob={.05,.1,.2,.5,.8,1.}})

module purge; module load bluebear
module load apps/python3/3.5.2
module load apps/keras/2.0.8-python-3.5.2

echo "starting "${ARR[${SLURM_ARRAY_TASK_ID}]}
python embedding/hyperbolic_embedding.py \
--data-directory ${DATA_DIR} --patience 1000 --lr 1. -b 32 ${ARR[${SLURM_ARRAY_TASK_ID}]} --context-size 3 --num-routing 0
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
