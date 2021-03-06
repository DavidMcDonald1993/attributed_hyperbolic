'''
Source: https://github.com/aditya-grover/node2vec/blob/master/src/node2vec.py
'''

import numpy as np
import scipy as sp
import networkx as nx
import random

class Graph():
	def __init__(self, graph, is_directed, p, q, jump_prob=0, feature_sim=None, seed=0):
		self.graph = graph
		self.is_directed = is_directed
		self.p = p
		self.q = q
		self.jump_prob = jump_prob
		self.feature_sim = feature_sim 
		np.random.seed(seed)
		random.seed(seed)

	def node2vec_walk(self, walk_length, start_node):
		'''
		Simulate a random walk starting from start node.
		'''
		graph = self.graph
		alias_nodes = self.alias_nodes
		alias_edges = self.alias_edges
		feature_sim = self.feature_sim

		jump = False

		walk = [start_node]

		while len(walk) < walk_length:
			cur = walk[-1]
			# node2vec style random walk 
			cur_nbrs = sorted(graph.neighbors(cur))

			if not (feature_sim[cur]==0).all() and self.jump_prob > 0 and (np.random.rand() < self.jump_prob or len(cur_nbrs) == 0):
				# random jump based on attribute similarity
				next_ = np.random.choice(len(feature_sim), replace=False, p=feature_sim[cur])
				walk.append(next_)
				jump = True

			elif len(cur_nbrs) > 0:
				if len(walk) == 1 or jump:
					walk.append(cur_nbrs[alias_draw(alias_nodes[cur][0], alias_nodes[cur][1])])
				else:
					prev = walk[-2]
					next_ = cur_nbrs[alias_draw(alias_edges[(prev, cur)][0], 
						alias_edges[(prev, cur)][1])]
					walk.append(next_)
				jump = False
			else:
				break

		return walk

	
	def simulate_walks(self, num_walks, walk_length):
		'''
		Repeatedly simulate random walks from each node.
		'''
		graph = self.graph
		walks = []
		nodes = list(graph.nodes())
		i = 0
		for walk_iter in range(num_walks):
			random.shuffle(nodes)
			for node in nodes:
				walks.append(self.node2vec_walk(walk_length=walk_length, start_node=node))
				if i % 1000 == 0:
					print ("performed walk {:05d}/{}".format(i, num_walks*len(graph)))
				i += 1

		return walks

	def get_alias_edge(self, src, dst):
		'''
		Get the alias edge setup lists for a given edge.
		'''
		graph = self.graph
		p = self.p
		q = self.q

		unnormalized_probs = []
		for dst_nbr in sorted(graph.neighbors(dst)):
			if dst_nbr == src:
				unnormalized_probs.append(graph[dst][dst_nbr]['weight']/p)
			elif graph.has_edge(dst_nbr, src):
				unnormalized_probs.append(graph[dst][dst_nbr]['weight'])
			else:
				unnormalized_probs.append(graph[dst][dst_nbr]['weight']/q)
		norm_const = sum(unnormalized_probs)
		normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

		return alias_setup(normalized_probs)


	def preprocess_transition_probs(self):
		'''
		Preprocessing of transition probabilities for guiding the random walks.
		'''
		print ("preprocessing transition probs")
		graph = self.graph
		is_directed = self.is_directed

		alias_nodes = {}
		i = 0
		for node in graph.nodes():
			unnormalized_probs = [graph[node][nbr]['weight'] for nbr in sorted(graph.neighbors(node))]
			norm_const = sum(unnormalized_probs)
			normalized_probs =  [float(u_prob)/norm_const for u_prob in unnormalized_probs]

			alias_nodes[node] = alias_setup(normalized_probs)
			if i % 1000 == 0:
				print ("preprocessed node {:04d}/{}".format(i, len(graph)))
			i += 1

		# triads = {}
		print ("preprocessed all nodes")
		self.alias_nodes = alias_nodes

		alias_edges = {}

		if is_directed:
			for edge in graph.edges():
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
		else:
			i = 0
			for edge in graph.edges():
				if i % 1000 == 0:
					print ("preprocessed edge {:05d}/{}".format(i, 2*len(graph.edges())))
				alias_edges[edge] = self.get_alias_edge(edge[0], edge[1])
				alias_edges[(edge[1], edge[0])] = self.get_alias_edge(edge[1], edge[0])
				i += 2

		print ("preprocessed all edges")

		self.alias_edges = alias_edges


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

def alias_draw(J, q):
	'''
	Draw sample from a non-uniform discrete distribution using alias sampling.
	'''
	K = len(J)

	kk = int(np.floor(np.random.rand()*K))
	if np.random.rand() < q[kk]:
		return kk
	else:
		return J[kk]

