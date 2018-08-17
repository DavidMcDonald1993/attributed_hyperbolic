#!/bin/bash

#SBATCH --job-name=coraJob
#SBATCH --output=coraJob_%A_%a.out
#SBATCH --error=coraJob_%A_%a.err
#SBATCH --array=1-16
#SBATCH --time=24:00:00
#SBATCH --ntasks=4
#SBATCH --mem-per-cpu=16gb


module purge; module load bluebear
module load bear-apps/2018a
module load Python/3.6.3-iomkl-2018a
module load apps/scikit-learn/0.19.0-python-3.5.2
module load TensorFlow/1.8.0-foss-2018a-Python-3.6.3
module load apps/keras/2.0.8-python-3.5.2
module load matplotlib/2.1.1-iomkl-2018a-Python-3.6.3

