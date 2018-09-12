from __future__ import print_function

import sys
import os
import json
import random
import numpy as np
import scipy as sp
from scipy.sparse import csr_matrix
import pandas as pd
import networkx as nx
from networkx.readwrite import json_graph

import pickle as pkl

from sklearn.preprocessing import StandardScaler

def load_karate(args):

	_dir = os.path.join(args.data_directory, "karate")

	topology_graph = nx.read_edgelist(os.path.join(_dir, "karate.edg"))

	label_df = pd.read_csv(os.path.join(_dir, "mod-based-clusters.txt"), sep=" ", index_col=0, header=None,)
	label_df.index = [str(idx) for idx in label_df.index]
	label_df = label_df.reindex(topology_graph.nodes())

	labels = label_df.iloc[:,0].values

	topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
	nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)

	features = np.genfromtxt(os.path.join(_dir, "/data/karate/feats.csv"), delimiter=",")

	return topology_graph, features, labels

def load_g2g_datasets(dataset_str, args):

	"""Load a graph from a Numpy binary file.
	Parameters
	----------
	file_name : str
		Name of the file to load.
	Returns
	-------
	graph : dict
		Dictionary that contains:
			* 'A' : The adjacency matrix in sparse matrix format
			* 'X' : The attribute matrix in sparse matrix format
			* 'z' : The ground truth class labels
			* Further dictionaries mapping node, class and attribute IDs
	"""
	file_name = os.path.join(args.data_directory,"g2g_datasets")
	file_name = os.path.join(file_name, dataset_str)
	if not file_name.endswith('.npz'):
		file_name += '.npz'
	with np.load(file_name) as loader:
		loader = dict(loader)
		A = sp.sparse.csr_matrix((loader['adj_data'], loader['adj_indices'],
						   loader['adj_indptr']), shape=loader['adj_shape'])

		X = sp.sparse.csr_matrix((loader['attr_data'], loader['attr_indices'],
						   loader['attr_indptr']), shape=loader['attr_shape'])

		z = loader.get('labels')

		graph = {
			'A': A,
			'X': X,
			'z': z
		}

		idx_to_node = loader.get('idx_to_node')
		if idx_to_node:
			idx_to_node = idx_to_node.tolist()
			graph['idx_to_node'] = idx_to_node

		idx_to_attr = loader.get('idx_to_attr')
		if idx_to_attr:
			idx_to_attr = idx_to_attr.tolist()
			graph['idx_to_attr'] = idx_to_attr

		idx_to_class = loader.get('idx_to_class')
		if idx_to_class:
			idx_to_class = idx_to_class.tolist()
			graph['idx_to_class'] = idx_to_class

	if args.directed:
		create_using = nx.DiGraph()
	else:
		create_using = nx.Graph() 

	topology_graph = nx.from_scipy_sparse_matrix(A, create_using=create_using)
	features = X.A
	labels = z

	if idx_to_class:
		label_info = graph["idx_to_class"]
	else: 
		label_info = None 

	return topology_graph, features, labels, label_info

def load_labelled_attributed_network(dataset_str, args, scale=False):
	"""Load data."""

	def parse_index_file(filename):
		"""Parse index file."""
		index = []
		for line in open(filename):
			index.append(int(line.strip()))
		return index

	def sample_mask(idx, l):
		"""Create mask."""
		mask = np.zeros(l)
		mask[idx] = 1
		return np.array(mask, dtype=np.bool)

	_dir = os.path.join(args.data_directory, "labelled_attributed_networks")

	names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
	objects = []
	for i in range(len(names)):
		with open(os.path.join(_dir, "ind.{}.{}".format(dataset_str, names[i])), 'rb') as f:
			if sys.version_info > (3, 0):
				objects.append(pkl.load(f, encoding='latin1'))
			else:
				objects.append(pkl.load(f))

	x, y, tx, ty, allx, ally, graph = tuple(objects)
	test_idx_reorder = parse_index_file(os.path.join(_dir, "ind.{}.test.index".format(dataset_str)))
	test_idx_range = np.sort(test_idx_reorder)

	if dataset_str == 'citeseer':
		scale=True
		# Fix citeseer dataset (there are some isolated nodes in the graph)
		# Find isolated nodes, add them as zero-vecs into the right position
		test_idx_range_full = list(range(min(test_idx_reorder), max(test_idx_reorder)+1))
		tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
		tx_extended[test_idx_range-min(test_idx_range), :] = tx
		tx = tx_extended
		ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
		ty_extended[test_idx_range-min(test_idx_range), :] = ty
		ty = ty_extended

	features = sp.sparse.vstack((allx, tx)).tolil()
	features[test_idx_reorder, :] = features[test_idx_range, :]
	adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

	labels = np.vstack((ally, ty))
	labels[test_idx_reorder, :] = labels[test_idx_range, :]
	labels = labels.argmax(axis=-1)

	# test_label_idx = test_idx_range.tolist()
	# train_label_idx = list(range(len(y)))
	# val_label_idx = list(range(len(y), len(y)+500))

	topology_graph = nx.from_numpy_matrix(adj.toarray())
	topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
	nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)

	if args.only_lcc:
		topology_graph = max(nx.connected_component_subgraphs(topology_graph), key=len)
		features = features[topology_graph.nodes()]
		labels = labels[topology_graph.nodes()]
		topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
		nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)

	# node2vec_graph = Graph(nx_G=G, is_directed=False, p=1, q=1)
	# node2vec_graph.preprocess_transition_probs()
	# walks = node2vec_graph.simulate_walks(num_walks=10, walk_length=5)

	# all_edges = G.edges()
	features = features.A
	if scale:
		scaler = StandardScaler()
		features = scaler.fit_transform(features)

	# feature_graph = create_feature_graph(features)

	return topology_graph, features, labels

def load_tf_interaction(args, normalize=True):
	_dir = os.path.join(args.data_directory, "tissue_classification")
	interaction_df = pd.read_csv(os.path.join(_dir, "NIHMS177825-supplement-03-1.csv"), 
		sep=",", skiprows=1).iloc[1:]
	topology_graph = nx.from_pandas_dataframe(interaction_df, "Gene 1 Symbol", "Gene 2 Symbol")

	features_df = pd.read_csv(os.path.join(_dir, "NIHMS177825-supplement-06-2.csv"), 
        sep=",", skiprows=1, index_col="Symbol", ).iloc[:,2:]

	# remove nodes with no expression data
	for n in topology_graph.nodes():
		if n not in features_df.index:
			topology_graph.remove_node(n)

	# sort features by node order
	features_df = features_df.loc[topology_graph.nodes(),:]

	features = features_df.values

	if normalize:
		features = StandardScaler().fit_transform(features)

	topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
	nx.set_edge_attributes(topology_graph, "weight", 1)
	labels = None
	label_info = None

	# print (len(topology_graph))
	# print (features)

	# raise SystemExit

	return topology_graph, features, labels, label_info


def load_ppi(args, normalize=True,):
	prefix = os.path.join(args.data_directory, "/ppi/ppi")
	G_data = json.load(open(prefix + "-G.json"))
	topology_graph = json_graph.node_link_graph(G_data)
	if isinstance(topology_graph.nodes()[0], int):
		conversion = lambda n : int(n)
	else:
		conversion = lambda n : n

	if os.path.exists(prefix + "-feats.npy"):
		features = np.load(prefix + "-feats.npy")
	else:
		print("No features present.. Only identity features will be used.")
		features = None
	id_map = json.load(open(prefix + "-id_map.json"))
	id_map = {conversion(k):int(v) for k,v in id_map.items()}
	class_map = json.load(open(prefix + "-class_map.json"))
	if isinstance(list(class_map.values())[0], list):
		lab_conversion = lambda n : n
	else:
		lab_conversion = lambda n : int(n)

	class_map = {conversion(k):lab_conversion(v) for k,v in class_map.items()}

	## Remove all nodes that do not have val/test annotations
	## (necessary because of networkx weirdness with the Reddit data)
	broken_count = 0
	for node in topology_graph.nodes():
		if not 'val' in topology_graph.node[node] or not 'test' in topology_graph.node[node]:
			topology_graph.remove_node(node)
			broken_count += 1
	print("Removed {:d} nodes that lacked proper annotations due to networkx versioning issues".format(broken_count))

	## Make sure the graph has edge train_removed annotations
	## (some datasets might already have this..)
	print("Loaded data.. now preprocessing..")
	for edge in topology_graph.edges():
		if ( topology_graph.node[edge[0]]['val'] or  topology_graph.node[edge[1]]['val'] or
			 topology_graph.node[edge[0]]['test'] or  topology_graph.node[edge[1]]['test']):
			 topology_graph[edge[0]][edge[1]]['train_removed'] = True
		else:
			 topology_graph[edge[0]][edge[1]]['train_removed'] = False

	if normalize and not features is None:
		from sklearn.preprocessing import StandardScaler
		train_ids = np.array([id_map[n] 
							  for n in  topology_graph.nodes() 
							  if not topology_graph.node[n]['val'] 
							  and not topology_graph.node[n]['test']])
		train_feats = features[train_ids]
		scaler = StandardScaler()
		scaler.fit(train_feats)
		features = scaler.transform(features)
		
	labels = np.array([class_map[n] for n in topology_graph.nodes()])
	nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)
	
	assert args.only_lcc
	if args.only_lcc:
		topology_graph = max(nx.connected_component_subgraphs(topology_graph), key=len)
		features = features[topology_graph.nodes()]
		labels = labels[topology_graph.nodes()]
		topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
		nx.set_edge_attributes(G=topology_graph, name="weight", values=1.)

	return topology_graph, features, labels

def load_wordnet(args):

    '''
    testing link prediciton / reconstruction / lexical entailment
    '''

    topology_graph = nx.read_edgelist(os.path.join(args.data_directory, "wordnet/noun_closure.tsv", ))
    topology_graph = nx.convert_node_labels_to_integers(topology_graph, label_attribute="original_name")
    nx.set_edge_attributes(topology_graph, name="weight", values=1)

    features = None
    labels = None
    label_info = None
    
    return topology_graph, features, labels, label_info