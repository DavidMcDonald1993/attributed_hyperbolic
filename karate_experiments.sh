#!/bin/bash

n_job=8
# rm -r {models,plots,logs,walks}/karate/*

# perform walks
# parallel -j $n_job -q python embedding/hyperbolic_embedding.py --dataset karate \
# 	 {1} -b 32 --lr 0.01 --just-walks --seed {2} {3} ::: \
# 	--no-attributes --multiply-attributes --jump-prob={0.05,0.1,0.2,0.5,1.0} --alpha={0.05,0.1,0.2,0.5,1.0} ::: 0 :::\
# 	--evaluate-link-prediction --evaluate-class-prediction

# # # sigmoid and softmax loss
# parallel -j $n_job -q python embedding/hyperbolic_embedding.py  --dataset karate --dim {1} \
# 	{2} {3} -b 32 --lr 0.01 --no-load --seed {4} ::: {2,3,5} ::: --sigmoid --softmax ::: \
# 	--no-attributes --multiply-attributes --alpha={0.05,0.1,0.2,0.5,1.0} --jump-prob={0.05,0.1,0.2,0.5,1.0} ::: 0

# # hyprbolic distance loss 
parallel -j $n_job -q python embedding/hyperbolic_embedding.py --dataset karate --dim {1} -r {2} -t {3} \
	{4} -b 32 --lr 0.01 --no-load --seed {5} ::: {2,3,5} ::: {1,3,5} ::: {1,3,5} ::: \
	--no-attributes --multiply-attributes --alpha={0.05,0.1,0.2,0.5,1.0} --jump-prob={0.05,0.1,0.2,0.5,1.0} ::: 0

