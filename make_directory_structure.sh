#!/bin/bash

#SBATCH --job-name=makeDirectoryStructure
#SBATCH --output=makeDirectoryStructure.out
#SBATCH --error=makeDirectoryStructure.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

# set -e

datasets=(cora_ml citeseer pubmed cora)
dims=(002 003 005 010 025 050 100)
seeds=({000..29})
atts=(no_attributes multiply_attributes jump_prob={0.05,0.1,0.2})

rm -r {models,plots,test_results,walks,logs}

for dataset in "${datasets[@]}"; do

	for dim in "${dims[@]}"; do

		for seed in "${seeds[@]}"; do

			for att in "${atts[@]}"; do



				mkdir -p {models,plots,logs}/${dataset}/dim=${dim}/seed=${seed}/all_components/\
eval_{lp/no_non_edges,class_pred}/softmax_loss/${att}

				mkdir -p walks/${dataset}/seed=${seed}/all_components/{no_lp,eval_lp/no_non_edges}/softmax_loss/${att}

				mkdir -p test_results/${dataset}/dim=${dim}/all_components/eval_{lp/no_non_edges,class_pred}/softmax_loss/${att}

				touch test_results/${dataset}/dim=${dim}/all_components/eval_{lp/no_non_edges,class_pred}/softmax_loss/${att}/test_results.lock

				touch logs/${dataset}/dim=${dim}/seed=${seed}/all_components/eval_{lp/no_non_edges,class_pred}/softmax_loss/${att}/log.csv

			done

		done

	done

done



# mkdir -p {models,plots,logs}/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_lp/no_non_edges/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

# mkdir -p {models,plots,logs}/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_class_pred/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

# mkdir -p walks/{cora_ml,citeseer,pubmed,cora}/\
# seed={000..29}/all_components/no_lp/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

# mkdir -p walks/{cora_ml,citeseer,pubmed,cora}/\
# seed={000..29}/all_components/eval_lp/no_non_edges/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

# mkdir -p test_results/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/all_components/eval_lp/no_non_edges/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

# mkdir -p test_results/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/all_components/eval_class_pred/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}

# touch test_results/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/all_components/eval_lp/no_non_edges/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/test_results.lock

# touch test_results/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/all_components/eval_class_pred/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/test_results.lock

# touch logs/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_lp/no_non_edges/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/log.csv

# touch logs/{cora_ml,citeseer,pubmed,cora}/\
# dim={002,003,005,010,020,050,100}/seed={000..29}/all_components/eval_class_pred/softmax_loss/\
# {no_attributes,multiply_attributes,jump_prob={0.05,0.1,0.2}}/log.csv