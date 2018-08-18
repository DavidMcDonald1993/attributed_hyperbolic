#!/bin/bash

n_job=3
# rm -r {models,plots,logs,walks}/cora/*

# perform walks
# parallel -j $n_job -q python embedding/hyperbolic_embedding.py  --dataset cora \
# 	 {1} --seed {2} -b 32 --lr 0.01 --just-walks {3} ::: \
# 	--no-attributes --multiply-attributes --jump-prob={0.05,0.1,0.2,0.5,1.0} --alpha={0.05,0.1,0.2,0.5,1.0} ::: 0 ::: \
# 	--evaluate-link-prediction --evaluate-class-prediction
# echo "completed walks"

# # link prediction experiment dim 128
# parallel -j $n_job -q python embedding/hyperbolic_embedding.py  --dataset cora --dim 128 \
# 	 {1} --seed {2} --evaluate-link-prediction --no-load -r {3} -t {4} ::: \
# 	--no-attributes --multiply-attributes --jump-prob={0.05,0.1,0.2,0.5,1.0} ::: 0 ::: {1,3,5} ::: {1,3,5}
parallel -j $n_job -q python embedding/hyperbolic_embedding.py  --dataset cora --dim 128 \
	 {1} --seed {2} --evaluate-link-prediction --no-load --sigmoid  ::: \
	--no-attributes --multiply-attributes --jump-prob=0.05 ::: 0 

# echo "completed link prediction experiments"

# # classification  dim 32?
# parallel -j $n_job -q python embedding/hyperbolic_embedding.py  --dataset cora --dim 32 \
# 	 {1} --seed {2} -b 32 --lr 0.01 --no-load -r {3} -t {4} ::: \
# 	--no-attributes --multiply-attributes --jump-prob={0.05,0.1,0.2,0.5,1.0} ::: {0..5} ::: {1,3,5} ::: {1,3,5}
