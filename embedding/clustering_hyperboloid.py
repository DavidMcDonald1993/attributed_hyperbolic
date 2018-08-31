from __future__ import print_function

import os
import re
import argparse
import numpy as np
import networkx as nx
import pandas as pd

import matplotlib.pyplot as plt

import h5py

from sklearn.cluster import DBSCAN
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score

from data_utils import load_karate, load_g2g_datasets, load_ppi, load_tf_interaction
from tree import TopologyConstrainedTree

def minkowki_dot(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance(u, v):
	mink_dp = minkowki_dot(u, v)
	mink_dp = np.minimum(mink_dp, -(1 + 1e-32))
	return np.arccosh(-mink_dp)

def load_embedding(filename):
	with h5py.File(filename, 'r') as f:
		embedding = np.array(f.get("embedding_layer/embedding_layer/embedding:0"))
	return embedding

def perform_clustering(dists, eps):
	dbs = DBSCAN(metric="precomputed", eps=eps, n_jobs=-1)
	labels = dbs.fit_predict(dists)
	return labels

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def convert_module_to_directed_module(module, ranks):
	module = nx.Graph(module)
	idx = module.nodes()

	directed_modules = []

	for connected_component in nx.connected_component_subgraphs(module):
		nodes = connected_component.nodes()
		root = nodes[ranks[nodes].argmin()]
		directed_edges = [(u, v) if ranks[u] < ranks[v] else (v, u) for u ,v in connected_component.edges()]
		directed_module = nx.DiGraph(directed_edges)

		if len(directed_module) > 0:
			directed_modules += [directed_module]

	return directed_modules

def grow_forest(data_train, directed_modules, ranks):

	forest = []
	for directed_module in directed_modules:
		feats = directed_module.nodes()
		root = feats[ranks[feats].argmin()]
		tree = TopologyConstrainedTree(parent_index=None, index=root, g=directed_module, 
			data=data_train, depth=0, max_depth=np.inf, min_samples_split=2, min_neighbours=1)
		forest.append(tree)
	return forest

def plot_disk_embeddings(edges, poincare_embedding, modules,):

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges) 

	all_modules = sorted(set(modules))
	num_modules = len(all_modules)
	colors = np.random.rand(num_modules, 3)

	fig = plt.figure(figsize=[14, 7])
	
	ax = fig.add_subplot(111)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	u_emb = poincare_embedding[edges[:,0]]
	v_emb = poincare_embedding[edges[:,1]]
	plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)
	for i, m in enumerate(all_modules):
		idx = modules == m
		plt.scatter(poincare_embedding[idx,0], poincare_embedding[idx,1], s=10, 
			c=colors[i], label="module={}".format(m) if m > -1 else "noise", zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=4)

	# plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),ncol=3)
	plt.show()
	# plt.savefig(path)
	plt.close()

# def load_graph(filename):
# 	pass

# def load_features(filename):
# 	pass

def load_labels(filename, label_column="Cell Line", label_of_interest="Mesoderm"):
	label_df = pd.read_csv(filename, index_col=0)
	labels = np.array(label_df.loc[:,label_column]==label_of_interest, dtype=np.float)
	return labels


def parse_model_filename(args):

	dataset = args.dataset
	directory = "dim={}/seed={}/".format(args.embedding_dim, args.seed)

	if args.only_lcc:
		directory += "lcc/"
	else:
		directory += "all_components/"

	if args.evaluate_link_prediction:
		directory += "eval_lp/"
		if args.add_non_edges:
			directory += "add_non_edges/"
		else:
			directory += "no_non_edges/"
	elif args.evaluate_class_prediction:
		directory += "eval_class_pred/"
	else: 
		directory += "no_lp/"


	if args.softmax:
		directory += "softmax_loss/"
	elif args.sigmoid:
		directory += "sigmoid_loss/"
	elif args.euclidean:
		directory += "euclidean_loss/"
	else:
		directory += "hyperbolic_distance_loss/r={}_t={}/".format(args.r, args.t)


	
	if args.multiply_attributes:
		directory += "multiply_attributes/"
	elif args.alpha>0:
		directory += "add_attributes_alpha={}/".format(args.alpha, )
	elif args.jump_prob > 0:
		directory += "jump_prob={}/".format(args.jump_prob)
	else:
		directory += "no_attributes/"

	args.model_path = os.path.join(args.model_path, dataset)
	args.model_path = os.path.join(args.model_path, directory)

	saved_models = sorted([f for f in os.listdir(args.model_path) 
		if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	assert len(saved_models) > 0
	return os.path.join(args.model_path, saved_models[-1])

def parse_args():
	parser = argparse.ArgumentParser(description="Density-based clustering in hyperbolic space")

	parser.add_argument("--data-directory", dest="data_directory", type=str, default="/data/",
		help="The directory containing data files (default is '/data/').")

	parser.add_argument("--dataset", dest="dataset", type=str, default="karate",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is karate)")
	
	
	parser.add_argument("-e", dest="max_eps", type=float, default=0.5,
		help="maximum eps.")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	parser.add_argument("-r", dest="r", type=float, default=3.,
		help="Radius of hypercircle (default is 3).")
	parser.add_argument("-t", dest="t", type=float, default=1.,
		help="Steepness of logistic function (defaut is 1).")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 2).", default=2)

	parser.add_argument("--alpha", dest="alpha", type=float, default=0,
		help="weighting of attributes (default is 0).")


	# parser.add_argument("--second-order", action="store_true", 
	# 	help="Use this flag to use second order topological similarity information.")
	parser.add_argument("--no-attributes", action="store_true", 
		help="Use this flag to not use attributes.")
	# parser.add_argument("--add-attributes", action="store_true", 
	# 	help="Use this flag to add attribute sim to adj.")
	parser.add_argument("--multiply-attributes", action="store_true", 
		help="Use this flag to multiply attribute sim to adj.")
	parser.add_argument("--jump-prob", dest="jump_prob", type=float, default=0, 
		help="Probability of randomly jumping to a similar node when walking.")


	# parser.add_argument("--distance", dest="distance", action="store_true", 
	# 	help="Use this flag to use hyperbolic distance loss.")
	parser.add_argument("--sigmoid", dest="sigmoid", action="store_true", 
		help="Use this flag to use sigmoid loss.")
	parser.add_argument("--softmax", dest="softmax", action="store_true", 
		help="Use this flag to use softmax loss.")
	parser.add_argument("--euclidean", dest="euclidean", action="store_true", 
		help="Use this flag to use euclidean negative sampling loss.")

	parser.add_argument("--model", dest="model_path", default="models/", 
		help="path to save model after each epoch (default is 'models/)'.")

	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	parser.add_argument('--evaluate-class-prediction', action="store_true", help='flag to evaluate class prediction')
	parser.add_argument('--evaluate-link-prediction', action="store_true", help='flag to evaluate link prediction')

	parser.add_argument('--no-non-edges', action="store_true", help='flag to not add non edges to training graph')
	parser.add_argument('--add-non-edges', action="store_true", help='flag to add non edges to training graph')


	args = parser.parse_args()
	return args

def main():
	args = parse_args()

	dataset = args.dataset
	assert dataset == "tf_interaction"
	if dataset == "karate":
		# topology_graph, features, labels = load_karate(args)
		raise Exception
	# elif dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
	# 	# topology_graph, features, labels = load_labelled_attributed_network(dataset, args)
	# 	topology_graph, features, labels, label_info = load_g2g_datasets(dataset, args)
	# elif dataset == "ppi":
	# 	topology_graph, features, labels = load_ppi(args)
	elif dataset == "tf_interaction":
		topology_graph, features, labels, label_info = load_tf_interaction(args, normalize=False)
	else:
		raise Exception

	features = features.T

	label_filename = os.path.join(args.data_directory, "tissue_classification/cell_lines.csv")
	print ("label_filename={}".format(label_filename))
	labels = load_labels(label_filename)

	model_filename = parse_model_filename(args)
	print ("loading model from {}".format(model_filename))

	embedding = load_embedding(model_filename)
	poincare_embedding = hyperboloid_to_poincare_ball(embedding)
	ranks = np.sqrt(np.sum(np.square(poincare_embedding), axis=-1, keepdims=False))

	# klein_embedding = hyperboloid_to_klein(embedding)
	dists = hyperbolic_distance(embedding, embedding)



	sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
	split_train, split_test = next(sss.split(features, labels))

	data = np.column_stack([features, labels])
	data_train = data[split_train]
	data_test = data[split_test]

	# max_modules = 0
	best_eps = -1
	best_f1 = 0
	best_forest = []
	# best_modules = [-1] * dists.shape[0]
	# best_num_connected = 0
	for eps in np.arange(0.01, args.max_eps, 0.01):
		modules = perform_clustering(dists, eps)

		num_modules = len(set(modules)) - 1
		print ("discovered {} modules with eps = {}".format(num_modules, eps))

		print ("fraction nodes in modules: {}".format((modules > -1).sum() / float(len(modules))))

		num_connected = 0
		directed_modules = []
		for m in range(num_modules):
			idx = np.where(modules == m)[0]
			module = topology_graph.subgraph(idx)
			num_connected += nx.is_connected(module)
			directed_modules += convert_module_to_directed_module(module, ranks)
		print ("created {} directed_modules".format(len(directed_modules)))
		print ("number connected modules = {}".format(num_connected))

		if len(directed_modules) > 3:
			forest = grow_forest(data_train, directed_modules, ranks)
			prediction = np.array([t.predict(data_test) for t in forest])
			print (prediction)
			print (prediction.mean(axis=0))
			prediction = prediction.mean(axis=0) > 0.5
			print (prediction.astype(np.float))
			f1 = f1_score(data_test[:,-1], prediction, average="macro")
			print ("f1={}".format(f1), len(forest))
			if f1 > best_f1:
				print ("best f1={}".format(f1))
				best_f1 = f1
				best_eps = eps
				best_forest = forest
		print ()

		# if num_connected > best_num_connected:
		# 	best_num_connected= num_connected
		# 	max_modules = num_modules
		# 	best_eps = eps
		# 	best_modules = modules

	eps = best_eps
	modules = perform_clustering(dists, eps)

	# for m in range(max_modules):
	# 	idx = np.where(modules == m)[0]
	# 	module = topology_graph.subgraph(idx)
	# 	print ("module {} contrains {} nodes and {} edges and connected={}".format(m, 
	# 		len(module), len(module.edges()), nx.is_connected(module)))
	# 	# if len(module) < 50 and nx.is_connected(module):
	# 	convert_module_to_tree(dists, poincare_embedding, module)

	prediction = np.array([t.predict(data_test) for t in best_forest])
	print (prediction)
	print (prediction.mean(axis=0))
	prediction = prediction.mean(axis=0) > 0.5
	print (prediction.astype(np.float))
	print (data_test[:,-1])
	print ([len(t) for t in best_forest])

	print ("Best eps was {}, best f1={}".format(best_eps, best_f1))
	plot_disk_embeddings(topology_graph.edges(), poincare_embedding, modules)

if __name__ == "__main__":
	main()