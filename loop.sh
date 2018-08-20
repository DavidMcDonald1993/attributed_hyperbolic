#!/bin/bash

a=({--no-attributes,--multiply-attributes,--jump-prob=0.05}" "-r{1,3,5}" "-t{1,3,5})

echo ${a[*]}
echo ${#a[*]}
# python embedding/hyperbolic_embedding.py --dataset karate ${a[6]} --context-size 2