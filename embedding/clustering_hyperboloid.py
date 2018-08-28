from __future__ import print_function

import os
import re
import argparse
import numpy as np
import networkx as nx

import matplotlib.pyplot as plt

import h5py

from sklearn.cluster import DBSCAN

from data_utils import load_karate, load_g2g_datasets, load_ppi

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

def convert_module_to_tree(dists, poincare_embedding, module):
	module = nx.Graph(module)
	idx = module.nodes()
	module_embedding = poincare_embedding[idx]
	t = np.sqrt(np.square(module_embedding).sum(axis=-1, keepdims=False))

	weights = {( u, v) : dists[u,v] for u, v in module.edges()} 
	nx.set_edge_attributes(module, "weight", weights)
	module = nx.convert_node_labels_to_integers(module)


	# pos = nx.spring_layout(module)
	pos = module_embedding[:,:2]
	nx.draw_networkx_nodes(module, pos, cmap=plt.get_cmap('jet'), 
	                       node_size = 50)
	nx.draw_networkx_labels(module, pos)
	nx.draw_networkx_edges(module, pos, arrows=False)
	plt.show()

	t_sort = t.argsort()
	i = 0
	root = t_sort[i]
	neighbors = module.neighbors(root)
	while len(neighbors) == 0:
		i += 1
		root = t_sort[i]
		neighbors = module.neighbors(root)
	print (i, neighbors)


	directed_edges = [(u, v) for u ,v in module.edges() if t[u] < t[v]]
	module = nx.DiGraph(directed_edges, )

	# module = nx.minimum_spanning_tree(module)
	nx.draw_networkx_nodes(module, pos, cmap=plt.get_cmap('jet'), 
	                       node_size = 50)
	nx.draw_networkx_labels(module, pos)
	nx.draw_networkx_edges(module, pos, arrows=True)
	plt.show()
	raise SystemExit

def plot_disk_embeddings(edges, poincare_embedding, clusters,):

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges) 

	all_clusters = sorted(set(clusters))
	num_clusters = len(all_clusters)
	colors = np.random.rand(num_clusters, 3)

	fig = plt.figure(figsize=[14, 7])
	
	ax = fig.add_subplot(111)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	u_emb = poincare_embedding[edges[:,0]]
	v_emb = poincare_embedding[edges[:,1]]
	plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)
	for i, c in enumerate(all_clusters):
		idx = clusters == c
		plt.scatter(poincare_embedding[idx,0], poincare_embedding[idx,1], s=10, 
			c=colors[i], label="cluster={}".format(c) if c > -1 else "noise", zorder=1)
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

def parse_filename(args):

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
	if dataset == "karate":
		topology_graph, features, labels = load_karate(args)
	elif dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
		# topology_graph, features, labels = load_labelled_attributed_network(dataset, args)
		topology_graph, features, labels, label_info = load_g2g_datasets(dataset, args)
	elif dataset == "ppi":
		topology_graph, features, labels = load_ppi(args)
	else:
		raise Exception

	filename = parse_filename(args)

	print ("loading model from {}".format(filename))

	embedding = load_embedding(filename)
	poincare_embedding = hyperboloid_to_poincare_ball(embedding)
	# klein_embedding = hyperboloid_to_klein(embedding)
	dists = hyperbolic_distance(embedding, embedding)
	max_modules = 0
	best_eps = -1
	best_modules = [-1] * dists.shape[0]
	best_num_connected = 0
	for eps in np.arange(0.01,args.max_eps,0.01):
		modules = perform_clustering(dists, eps)

		num_modules = len(set(modules)) - 1
		print ("discovered {} modules with eps = {}".format(num_modules, eps))

		print ("fraction nodes in modules: {}".format((modules > -1).sum() / float(len(modules))))

		num_connected = 0
		for m in range(num_modules):
			idx = np.where(modules == m)[0]
			module = topology_graph.subgraph(idx)
			num_connected += nx.is_connected(module)
		print ("number connected modules = {}".format(num_connected))

		if num_modules > 0:
			ratio = float(num_connected) / num_modules

		if num_connected > best_num_connected:
			best_num_connected= num_connected
			max_modules = num_modules
			best_eps = eps
			best_modules = modules

	eps = best_eps
	modules = perform_clustering(dists, eps)

	for m in range(max_modules):
		idx = np.where(modules == m)[0]
		module = topology_graph.subgraph(idx)
		print ("module {} contrains {} nodes and {} edges and connected={}".format(m, 
			len(module), len(module.edges()), nx.is_connected(module)))
		# if len(module) < 50 and nx.is_connected(module):
		convert_module_to_tree(dists, poincare_embedding, module)


	print ("Best eps was {}, found {} clusters, {} connected".format(best_eps, max_modules, best_num_connected))
	plot_disk_embeddings(topology_graph.edges(), poincare_embedding, best_modules)

if __name__ == "__main__":
	main()