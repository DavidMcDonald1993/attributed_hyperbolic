!/bin/bash

#SBATCH --job-name=noGPUexp
#SBATCH --output=noGPUexp_%A_%a.out
#SBATCH --error=noGPULexp_%A_%a.err
#SBATCH --array=0-1439
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

DATA_DIR="/rds/homes/d/dxm237/data"

ARR=(--dataset={cora_ml,citeseer}" "--seed={0..29}" "--softmax" "--dim={5,10,25}" --evaluate-"{link,class}"-prediction "--{no-attributes,jump-prob={.05,.1,.2}})

module purge; module load bluebear
module load apps/cuda/8.0.44
module load apps/cudnn/6.0
-I${CUDNN_ROOT}/include/cudnn.h -L${CUDNN_ROOT}/lib64/libcudnn.so
module load apps/python3/3.5.2
module load apps/tensorflow/1.3.1-python-3.5.2
module load apps/keras/2.0.8-python-3.5.2

echo "starting "${ARR[${SLURM_ARRAY_TASK_ID}]}
python embedding/hyperbolic_embedding.py \
--data-directory ${DATA_DIR} --patience 1000 --lr .3 -b 512 -e 300 --no-load ${ARR[${SLURM_ARRAY_TASK_ID}]} --context-size 3
echo "completed "${ARR[${SLURM_ARRAY_TASK_ID}]}
