import argparse

import numpy as np

from embedding.data_utils import load_g2g_datasets, load_ppi
from embedding.visualise import plot_degree_dist

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Plot degree dists script")

	parser.add_argument("--data-directory", dest="data_directory", type=str, default="/data/",
		help="The directory containing data files (default is '/data/').")

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')


	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	args = parser.parse_args()
	return args

def main():

	args = parse_args()

	datasets = ["cora_ml", "citeseer", "pubmed", "ppi"]

	f, a = plt.subplots(nrows=1, ncols=len(datasets), figsize=(5*len(datasets), 5), dpi= 80)

	for i, dataset in enumerate(datasets):

		if dataset in ["cora_ml", "citeseer", "pubmed"]:
			load = load_g2g_datasets
		else:
			load = load_ppi

		graph, _, _ = load(dataset, args)
		plot_degree_dist(dataset, graph, a[i])

	plt.savefig("degree_dist_plot.png")
	plt.close()



if __name__ == "__main__":
	main()