#!/bin/bash

# datasets={cora_ml,citeseer,pubmed,cora}
# dim={002,003,005,010,025,050,100}
# seed={000..29}
# att={no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

rm -r {models,plots,test_results,walks,logs}

mkdir -p {models,plots,logs}/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_lp/no_non_edges/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

mkdir -p {models,plots,logs}/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_class_pred/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

mkdir -p walks/{cora_ml,citeseer,pubmed,cora}/\
seed={000..29}/all_components/no_lp/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

mkdir -p walks/{cora_ml,citeseer,pubmed,cora}/\
seed={000..29}/all_components/eval_lp/no_non_edges/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

mkdir -p test_results/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/all_components/eval_lp/no_non_edges/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

mkdir -p test_results/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/all_components/eval_class_pred/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

touch test_results/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/all_components/eval_lp/no_non_edges/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/test_results.lock

touch test_results/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/all_components/eval_class_pred/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/test_results.lock

touch logs/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_lp/no_non_edges/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/log.csv

touch logs/{cora_ml,citeseer,pubmed,cora}/\
dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_class_pred/softmax_loss/\
{no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/log.csv