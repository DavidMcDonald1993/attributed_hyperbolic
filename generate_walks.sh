#!/bin/bash

#SBATCH --job-name=generateWalks
#SBATCH --output=generateWalks_%A_%a.out
#SBATCH --error=generateWalks_%A_%a.err
#SBATCH --array=0-35
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=64gb


PROJECT_DIR=/rds/projects/2018/hesz01/attributed_hyperbolic
arr=(--seed={0..2}" "--dataset={cora_ml,cora,citeseer,pubmed}" "{--evaluate-class-prediction,--evaluate-link-prediction" "{--no-non-edges,--add-non-edges}}" "--multiply-attributes)

# echo ${#arr[*]}

module purge; module load bluebear
module load apps/cuda/8.0.44
module load apps/cudnn/6.0
-I${CUDNN_ROOT}/include/cudnn.h -L${CUDNN_ROOT}/lib64/libcudnn.so
# module load bear-apps/2018a
# module load Python/3.6.3-iomkl-2018a
module load apps/python2/2.7.11
# module load apps/python3/3.5.2
# module load apps/scikit-learn/0.19.0-python-3.5.2
# module load apps/h5py/2.7.0-python-3.5.2
# module load TensorFlow/1.8.0-foss-2018a-Python-3.6.3
module load apps/tensorflow/1.3.1-python-2.7.11
# module load apps/tensorflow/1.3.1-python-3.5.2-cuda-8.0.44
# module load apps/tensorflow/1.3.1-python-2.7.11-cuda-8.0.44
module load apps/keras/2.0.8-python-2.7.11
# module load apps/keras/2.0.8-python-3.5.2-cuda-8.0.44

echo starting, ${arr[${SLURM_ARRAY_TASK_ID}]}
python embedding/hyperbolic_embedding.py  --data-directory ${PROJECT_DIR}/data \
${arr[${SLURM_ARRAY_TASK_ID}]} --evaluate-link-prediction --just-walks