#!/bin/bash

#SBATCH --job-name=coraJob
#SBATCH --output=coraJob_%A_%a.out
#SBATCH --error=coraJob_%A_%a.err
#SBATCH --array=0-0
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16gb

arr=(--no-attributes --multiply-attributes --jump-prob=0.05)


module purge; module load bluebear
module load bear-apps/2018a
module load apps/python3/3.5.2
module load apps/scikit-learn/0.19.0-python-3.5.2
module load apps/h5py/2.7.0-python-3.5.2
module load TensorFlow/1.8.0-foss-2018a-Python-3.6.3
module load apps/keras/2.0.8-python-3.5.2
module load matplotlib/2.1.1-iomkl-2018a-Python-3.6.3

echo done
# python embedding/hyperbolic_embedding.py  --dataset cora --dim 32 \
# 	--evaluate-link-prediction --no-load --sigmoid ${arr[${SLURM_ARRAY_TASK_ID}]}