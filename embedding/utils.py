from __future__ import print_function

import os
import numpy as np
import networkx as nx
from node2vec_sampling import Graph 

import random

from sklearn.metrics.pairwise import cosine_similarity

def alias_setup(probs):
	'''
	Compute utility lists for non-uniform sampling from discrete distributions.
	Refer to https://hips.seas.harvard.edu/blog/2013/03/03/the-alias-method-efficient-sampling-with-many-discrete-outcomes/
	for details
	'''
	K = len(probs)
	q = np.zeros(K)
	J = np.zeros(K, dtype=np.int)

	smaller = []
	larger = []
	for kk, prob in enumerate(probs):
		q[kk] = K*prob
		if q[kk] < 1.0:
			smaller.append(kk)
		else:
			larger.append(kk)

	while len(smaller) > 0 and len(larger) > 0:
		small = smaller.pop()
		large = larger.pop()

		J[small] = large
		q[large] = q[large] + q[small] - 1.0
		if q[large] < 1.0:
			smaller.append(large)
		else:
			larger.append(large)

	return J, q

def alias_draw(J, q, size=1):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = np.floor(np.random.uniform(high=K, size=size)).astype(np.int)
	r = np.random.uniform(size=size)
	idx = r >= q[kk]
	kk[idx] = J[kk[idx]]
	return kk

def convert_edgelist_to_dict(edgelist, undirected=True, self_edges=False):
	if edgelist is None:
		return None
	sorts = [lambda x: sorted(x)]
	if undirected:
		sorts.append(lambda x: sorted(x, reverse=True))
	edges = (sort(edge) for edge in edgelist for sort in sorts)
	edge_dict = {}
	for u, v in edges:
		if self_edges:
			default = [u]#set(u)
		else:
			default = []#set()
		edge_dict.setdefault(u, default).append(v)
	# for u, v in edgelist:
	# 	assert v in edge_dict[u]
	# 	if undirected:
	# 		assert u in edge_dict[v]
	return edge_dict

def get_training_sample(batch_positive_samples, negative_samples, num_negative_samples, alias_dict):

	input_nodes = batch_positive_samples[:,0]

	batch_negative_samples = np.array([
		# np.random.choice(negative_samples[u], 
		# replace=True, size=(num_negative_samples,), 
		# p=probs[u] if probs is not None else probs
		# )
		negative_samples[u][alias_draw(alias_dict[u][0], alias_dict[u][1], num_negative_samples)]
		for u in input_nodes
	], dtype=np.int64)
	batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
	return batch_nodes


def make_validation_data(edges, non_edge_dict, args):

	edges = np.array(edges)
	idx = np.random.choice(len(edges), size=min(len(edges), args.batch_size), replace=False,)
	positive_samples = edges[idx]#
	# non_edge_dict = convert_edgelist_to_dict(non_edges)
	negative_samples = np.array([
		np.random.choice(non_edge_dict[u], size=args.num_negative_samples, replace=True,)
		for u in positive_samples[:,0]
	])
	x = np.append(positive_samples, negative_samples, axis=-1)
	# x = get_training_sample(positive_samples, 
	# 	non_edge_dict, args.num_negative_samples, probs=None)
	y = np.zeros(list(x.shape)+[1], dtype=np.int64)

	return x, y


def create_second_order_topology_graph(topology_graph, args):

	adj = nx.adjacency_matrix(topology_graph).A
	adj_sim = cosine_similarity(adj)
	adj_sim -= np.identity(len(topology_graph))
	adj_sim [adj_sim  < args.rho] = 0
	second_order_topology_graph = nx.from_numpy_matrix(adj_sim)

	print ("Created second order topology graph graph with {} edges".format(len(second_order_topology_graph.edges())))

	return second_order_topology_graph


def create_feature_graph(features, args):

	features_sim = cosine_similarity(features)
	features_sim -= np.identity(len(features))
	features_sim [features_sim  < args.rho] = 0
	feature_graph = nx.from_numpy_matrix(features_sim)

	print ("Created feature correlation graph with {} edges".format(len(feature_graph.edges())))

	return feature_graph

def split_edges(edges, non_edges, args, val_split=0.05, test_split=0.1, neg_mul=1):
	
	num_val_edges = int(np.ceil(len(edges) * val_split))
	num_test_edges = int(np.ceil(len(edges) * test_split))

	random.seed(args.seed)
	random.shuffle(edges)

	val_edges = edges[:num_val_edges]
	test_edges = edges[num_val_edges:num_val_edges+num_test_edges]
	train_edges = edges[num_val_edges+num_test_edges:]

	val_non_edges = non_edges[:num_val_edges*neg_mul]
	test_non_edges = non_edges[num_val_edges*neg_mul:num_val_edges*neg_mul+num_test_edges*neg_mul]

	return train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges)

def determine_positive_and_negative_samples(nodes, walks, context_size):

	print ("determining positive and negative samples")

	if not isinstance(nodes, set):
		nodes = set(nodes)
	
	all_positive_samples = {n: {n} for n in sorted(nodes)}
	positive_samples = []

	counts = {n: 0. for n in sorted(nodes)}

	for num_walk, walk in enumerate(walks):
		for i in range(len(walk)):
			u = walk[i]
			counts[u] += 1	
			for j in range(context_size):
			# for j in range(i+1, min(len(walk), i+1+context_size)):
				if i+j+1 >= len(walk):
					continue
				v = walk[i+j+1]
				if u == v:
					continue
				# n = 1
				n = context_size - j
				positive_samples.extend([(u, v)] * n)
				positive_samples.extend([(v, u)] * n)
				
				all_positive_samples[u].add(v)
				all_positive_samples[v].add(u)

 
		if num_walk % 1000 == 0:  
			print ("processed walk {}/{}".format(num_walk, len(walks)))
	print ("processed walk {}/{}".format(num_walk, len(walks)))
		

	negative_samples = {n: np.array(sorted(nodes.difference(all_positive_samples[n]))) for n in sorted(nodes)}
	for u in negative_samples:
		assert u not in negative_samples[u], "u should not be in negative samples"
		assert len(negative_samples[u]) > 0, "node {} does not have any negative samples".format(u)

	print ("DETERMINED POSITIVE AND NEGATIVE SAMPLES")
	print ("found {} positive sample pairs".format(len(positive_samples)))

	counts = np.array(list(counts.values())) ** 0.75
	probs = counts #/ counts.sum()

	prob_dict = {n: probs[negative_samples[n]] for n in sorted(nodes)}
	prob_dict = {n: probs / probs.sum() for n, probs in prob_dict.items()}
	# for k, v in prob_dict.items():
		# print (k, len(negative_samples[k]), len(v), v.sum())
		# print (probs)
		# print (alias_setup(probs))
	# print (np.array(prob_dict.values()).shape)
	# prob_dict = {n: np.ones_like(negative_samples[n], dtype=np.float) / len(negative_samples[n]) for n in prob_dict.keys()}
	# probs = {n: counts[negative_samples[n]] / counts[negative_samples[n]].sum() for n in sorted(nodes)}
	alias_dict = {n: alias_setup(probs) for n, probs in prob_dict.items()}

	print ("PREPROCESSED NEGATIVE SAMPLE PROBS")

	return positive_samples, negative_samples, alias_dict

def load_walks(G, walk_file, feature_sim, args):

	def save_walks_to_file(walks, walk_file):
		with open(walk_file, "w") as f:
			for walk in walks:
				f.write(",".join([str(n) for n in walk]) + "\n")

	def load_walks_from_file(walk_file, walk_length):

		walks = []

		with open(walk_file, "r") as f:
			for line in (line.rstrip() for line in f.readlines()):
				walks.append([int(n) for n in line.split(",")])
		return walks


	if not os.path.exists(walk_file):
		node2vec_graph = Graph(nx_G=G, is_directed=False, p=args.p, q=args.q,
			jump_prob=args.jump_prob, feature_sim=feature_sim, seed=args.seed)
		node2vec_graph.preprocess_transition_probs()
		walks = node2vec_graph.simulate_walks(num_walks=args.num_walks, walk_length=args.walk_length)
		save_walks_to_file(walks, walk_file)
		print ("saved walks to {}".format(walk_file))

	else:
		print ("loading walks from {}".format(walk_file))
		walks = load_walks_from_file(walk_file, args.walk_length)
	return walks
