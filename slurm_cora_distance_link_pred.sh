#!/bin/bash

arr=({--no-attributes,--multiply-attributes,--jump-prob=0.05}" "-r{1,3,5}" "-t{1,3,5})

#SBATCH --job-name=coraDistanceLinkPred
#SBATCH --output=coraDistanceLinkPred_%A_%a.out
#SBATCH --error=coraDistanceLinkPred_%A_%a.err
#SBATCH --array=0-${#arr[*]}
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=3
#SBATCH --mem-per-cpu=16gb


echo ${SLURM_ARRAY_TASK_MAX}


module purge; module load bluebear
# module load bear-apps/2018a
# module load Python/3.6.3-iomkl-2018a
module load apps/python2/2.7.11
# module load apps/scikit-learn/0.19.0-python-3.5.2
# module load apps/h5py/2.7.0-python-3.5.2
# module load TensorFlow/1.8.0-foss-2018a-Python-3.6.3
module load apps/tensorflow/1.3.1-python-2.7.11
module load apps/keras/2.0.8-python-2.7.11
# module load matplotlib/2.1.1-iomkl-2018a-Python-3.6.3
module load apps/cudnn/6.0
-I${CUDNN_ROOT}/include/cudnn.h -L${CUDNN_ROOT}/lib64/libcudnn.so

# echo done
python embedding/hyperbolic_embedding.py  --dataset cora --dim 128 --data-directory /rds/homes/d/dxm237/data \
--no-load ${arr[${SLURM_ARRAY_TASK_ID}]}