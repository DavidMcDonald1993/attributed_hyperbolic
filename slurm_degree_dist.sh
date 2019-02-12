#!/bin/bash

#SBATCH --job-name=degreeDist
#SBATCH --output=degreeDist.out
#SBATCH --error=degreeDist.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=100G
# SBATCH --mail-type ALL

DATA_DIR="/rds/homes/d/dxm237/data"

module purge; module load bluebear
module load apps/python3/3.5.2
python plot_degree_dists.py --data-directory ${DATA_DIR} 
