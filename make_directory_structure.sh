#!/bin/bash

#SBATCH --job-name=makeDirectoryStructure
#SBATCH --output=makeDirectoryStructure.out
#SBATCH --error=makeDirectoryStructure.err
#SBATCH --time=01:00:00
#SBATCH --ntasks=1
#SBATCH --mem=8G

# set -e

datasets=(cora_ml citeseer pubmed)
dims=(002 003 005 010 025 050)
seeds=({000..5})
atts=(no_attributes jump_prob={0.05,0.1,0.2,0.5,0.8,1.0})

rm -r {models,plots,test_results,samples,walks,logs}

for dataset in "${datasets[@]}"; do

	for dim in "${dims[@]}"; do

		for seed in "${seeds[@]}"; do

			for att in "${atts[@]}"; do

				mkdir -p {models,plots,logs}/${dataset}/dim=${dim}/seed=${seed}/all_components/\
eval_{lp,class_pred}/softmax_loss/${att}

				mkdir -p {walks,samples}/${dataset}/seed=${seed}/all_components/{no_lp,eval_lp}

				mkdir -p test_results/${dataset}/dim=${dim}/all_components/eval_{lp,class_pred}/softmax_loss/${att}

				touch test_results/${dataset}/dim=${dim}/all_components/eval_{lp,class_pred}/softmax_loss/${att}/test_results.lock

				touch logs/${dataset}/dim=${dim}/seed=${seed}/all_components/eval_{lp,class_pred}/softmax_loss/${att}/log.csv

			done

		done

	done

done

