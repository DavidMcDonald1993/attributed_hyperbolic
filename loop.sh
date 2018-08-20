#!/bin/bash

arr=(-r{5,3,1}" "-t{1,3,5}" "{--no-attributes,--multiply-attributes,--jump-prob=0.05})

echo ${arr[*]}
echo ${#arr[*]}
# python embedding/hyperbolic_embedding.py --dataset karate ${a[6]} --context-size 2