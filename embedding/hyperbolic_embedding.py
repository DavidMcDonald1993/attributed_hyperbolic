from __future__ import print_function


import os
import h5py
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ["PYTHON_EGG_CACHE"] = "/rds/projects/2018/hesz01/attributed_hyperbolic/python-eggs"
import multiprocessing 
import re
import argparse
import json
import sys

import random

import numpy as np
import networkx as nx
from scipy.sparse import identity
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics.pairwise import cosine_similarity, euclidean_distances

from data_utils import load_karate, load_labelled_attributed_network, load_ppi, load_g2g_datasets, load_tf_interaction, load_wordnet
from utils import load_walks, determine_positive_and_negative_samples, convert_edgelist_to_dict, split_edges, get_training_sample, make_validation_data, threadsafe_save_test_results
from callbacks import PeriodicStdoutLogger, hyperboloid_to_klein, hyperboloid_to_poincare_ball, hyperbolic_distance_hyperboloid_pairwise
from losses import hyperbolic_negative_sampling_loss, hyperbolic_sigmoid_loss, hyperbolic_softmax_loss, euclidean_negative_sampling_loss
from metrics import evaluate_rank_and_MAP, evaluate_classification
from callbacks import plot_disk_embeddings, plot_euclidean_embedding, plot_roc, plot_classification, plot_precisions_recalls
from generators import training_generator, TrainingSequence

from keras.layers import Input, Layer, Dense, Embedding
from keras.models import Model
from keras import backend as K
from keras.callbacks import Callback, TerminateOnNaN, TensorBoard, ModelCheckpoint, CSVLogger, EarlyStopping

import tensorflow as tf
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops, control_flow_ops
from tensorflow.python.training import optimizer

K.set_floatx("float64")
K.set_epsilon(1e-15)

np.set_printoptions(suppress=True)


# TensorFlow wizardry
config = tf.ConfigProto()

# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.5

config.log_device_placement=False
config.allow_soft_placement=True

# Create a session with the above options specified.
K.tensorflow_backend.set_session(tf.Session(config=config))


def minkowski_dot(x, y):
	# assert len(x.shape) == 2
	axes = len(x.shape) - 1, len(y.shape) -1
	return K.batch_dot( x[...,:-1], y[...,:-1], axes=axes) - K.batch_dot(x[...,-1:], y[...,-1:], axes=axes)


def hyperboloid_initializer(shape, r_max=1e-3):

	def poincare_ball_to_hyperboloid(X, append_t=True):
		x = 2 * X
		t = 1. + K.sum(K.square(X), axis=-1, keepdims=True)
		if append_t:
			x = K.concatenate([x, t], axis=-1)
		return 1 / (1. - K.sum(K.square(X), axis=-1, keepdims=True)) * x

	def sphere_uniform_sample(shape, r_max):
		num_samples, dim = shape
		X = tf.random_normal(shape=shape, dtype=K.floatx())
		X_norm = K.sqrt(K.sum(K.square(X), axis=-1, keepdims=True))
		U = tf.random_uniform(shape=(num_samples, 1), dtype=K.floatx())
		return r_max * U ** (1./dim) * X / X_norm

	w = sphere_uniform_sample(shape, r_max=r_max)
	return poincare_ball_to_hyperboloid(w)
	# return np.genfromtxt("../data/labelled_attributed_networks/cora-lcc-warmstart.weights")

class EmbeddingLayer(Layer):
	
	def __init__(self, num_nodes, embedding_dim, **kwargs):
		super(EmbeddingLayer, self).__init__(**kwargs)
		self.num_nodes = num_nodes
		self.embedding_dim = embedding_dim

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		self.embedding = self.add_weight(name='embedding', 
									  shape=(self.num_nodes, self.embedding_dim),
									  initializer=hyperboloid_initializer,
									  trainable=True)


		super(EmbeddingLayer, self).build(input_shape)

	def call(self, x):
		x = K.cast(x, dtype=tf.int64)
		embedding = tf.gather(self.embedding, x, name="embedding_gather")
		
		# embedding = K.dot(x, self.embedding, )

		return embedding

	def compute_output_shape(self, input_shape):
		return (input_shape[0], input_shape[1], self.embedding_dim+1)
	
	def get_config(self):
		base_config = super(EmbeddingLayer, self).get_config()
		return base_config.update({"num_nodes": self.num_nodes, "embedding_dim": self.embedding_dim})

class ExponentialMappingOptimizer(optimizer.Optimizer):
	
	def __init__(self, learning_rate=0.001, use_locking=False, name="ExponentialMappingOptimizer"):
		super(ExponentialMappingOptimizer, self).__init__(use_locking, name)
		self._lr = learning_rate

	def _prepare(self):
		self._lr_t = ops.convert_to_tensor(self._lr, name="learning_rate", dtype=K.floatx())

	def _apply_dense(self, grad, var):
		# print "dense"
		assert False
		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
			# K.floatx())
		spacial_grad = grad[:,:-1]
		t_grad = -grad[:,-1:]
		
		ambient_grad = tf.concat([spacial_grad, t_grad], axis=-1)
		tangent_grad = self.project_onto_tangent_space(var, ambient_grad)
		
		exp_map = self.exponential_mapping(var, - lr_t * tangent_grad)
		
		return tf.assign(var, exp_map)
		
	def project_onto_tangent_space(self, hyperboloid_point, minkowski_tangent):
		tang = minkowski_tangent + minkowski_dot(hyperboloid_point, minkowski_tangent) * hyperboloid_point
		return tang
   
	def exponential_mapping( self, p, x, ):

		def adjust_to_hyperboloid(x):
			x = x[:,:-1]
			t = K.sqrt(1. + K.sum(K.square(x), axis=-1, keepdims=True))
			return tf.concat([x, t], axis=-1)

		norm_x = K.sqrt( K.maximum(K.cast(0., K.floatx()), minkowski_dot(x, x), ) )

		#####################################################
		# exp_map_p = tf.cosh(norm_x) * p
		
		# idx = tf.cast( tf.where(norm_x > K.cast(0., K.floatx()), )[:,0], tf.int32)
		# non_zero_norm = tf.gather(norm_x, idx)
		# z = tf.gather(x, idx) / non_zero_norm

		# updates = tf.sinh(non_zero_norm) * z
		# dense_shape = tf.cast( tf.shape(p), tf.int64)
		# exp_map_x = tf.scatter_nd(indices=idx, updates=updates, shape=dense_shape)

		
		# exp_map = exp_map_p + exp_map_x    
		###################################################
		y = p
		# z = x / norm_x
		z = x / K.maximum(norm_x, K.epsilon())

		# norm_x = K.minimum(norm_x, .01)

		exp_map = tf.cosh(norm_x) * y + tf.sinh(norm_x) * z
		#####################################################
		exp_map = adjust_to_hyperboloid(exp_map)
		# idx = tf.where(tf.abs(exp_map + 1) < K.epsilon())[:,0]
		# params = tf.gather(exp_map, idx)

		# params = adjust_to_hyperboloid(params)
		# exp_map = tf.scatter_update(ref=exp_map, updates=params, indices=idx)

		return exp_map

	def _apply_sparse(self, grad, var):
		# assert False
		indices = grad.indices
		values = grad.values
		# dense_shape = grad.dense_shape
		# p = tf.nn.embedding_lookup(var, indices)
		p = tf.gather(var, indices, name="gather_apply_sparse")

		lr_t = math_ops.cast(self._lr_t, var.dtype.base_dtype)
		spacial_grad = values[:,:-1]
		t_grad = -values[:,-1:]

		ambient_grad = tf.concat([spacial_grad, t_grad], axis=-1, name="optimizer_concat")
		tangent_grad = self.project_onto_tangent_space(p, ambient_grad)
		# exp_map = ambient_grad
		exp_map = self.exponential_mapping(p, - lr_t * tangent_grad)

		out = tf.scatter_update(ref=var, updates=exp_map, indices=indices, name="scatter_update")

		# return control_flow_ops.group(out, name="grouping")
		return out
		# return tf.assign(var, var)

def build_model(num_nodes, args):

	x = Input(shape=(1+args.num_positive_samples+args.num_negative_samples,), name="model_input")
	if args.euclidean:
		y = Embedding(num_nodes, args.embedding_dim, input_length=1+args.num_positive_samples+args.num_negative_samples,
			embeddings_regularizer=None, name="embedding_layer")(x)
	else:
		y = EmbeddingLayer(num_nodes, args.embedding_dim, name="embedding_layer")(x)
	# y = Dense(args.embedding_dim, use_bias=False, activation=None, 
	#   kernel_initializer=hyperboloid_initializer, name="embedding_layer")(x)

	model = Model(x, y)


	# if os.path.exists(args.model_path):
	#   print ("Loading model from file: {}".format(args.model_path))
	#   model.load_weights(args.model_path)

	initial_epoch = 0

	saved_models = sorted([f for f in os.listdir(args.model_path) 
		if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	# initial_epoch = len(saved_models)

	# print (model.layers[-1].get_weights()[0])

	if len(saved_models) > 0:

		model_file = os.path.join(args.model_path, saved_models[-1])
		print ("Loading model from file: {}".format(model_file))
		model.load_weights(model_file)

		print (model.layers[-1].get_weights()[0])

		initial_epoch = int(saved_models[-1].split(".")[0])
		print ("initial epoch={}".format(initial_epoch))

	return model, initial_epoch

def load_embedding(filename):
	with h5py.File(filename, 'r') as f:
		embedding = np.array(f.get("embedding_layer/embedding_layer/embedding:0"))
	return embedding

def parse_args():
	'''
	parse args from the command line
	'''
	parser = argparse.ArgumentParser(description="Hyperbolic Skipgram for feature learning on complex networks")

	parser.add_argument("--data-directory", dest="data_directory", type=str, default="/data/",
		help="The directory containing data files (default is '/data/').")

	parser.add_argument("--dataset", dest="dataset", type=str, default="karate",
		help="The dataset to load. Must be one of [wordnet, cora, citeseer, pubmed,\
		AstroPh, CondMat, GrQc, HepPh, karate]. (Default is karate)")

	parser.add_argument("--seed", dest="seed", type=int, default=0,
		help="Random seed (default is 0).")

	parser.add_argument("-r", dest="r", type=float, default=3.,
		help="Radius of hypercircle (default is 3).")
	parser.add_argument("-t", dest="t", type=float, default=1.,
		help="Steepness of logistic function (defaut is 1).")


	parser.add_argument("--lr", dest="lr", type=float, default=5e-2,
		help="Learning rate (default is 5e-2).")

	parser.add_argument("--rho", dest="rho", type=float, default=0,
		help="Minimum feature correlation (default is 0).")

	parser.add_argument("-e", "--num_epochs", dest="num_epochs", type=int, default=50000,
		help="The number of epochs to train for (default is 50000).")
	parser.add_argument("-b", "--batch_size", dest="batch_size", type=int, default=512, 
		help="Batch size for training (default is 512).")
	parser.add_argument("--nneg", dest="num_negative_samples", type=int, default=10, 
		help="Number of negative samples for training (default is 10).")
	parser.add_argument("--context-size", dest="context_size", type=int, default=1,
		help="Context size for generating positive samples (default is 1).")
	parser.add_argument("--patience", dest="patience", type=int, default=25,
		help="The number of epochs of no improvement in validation loss before training is stopped. (Default is 25)")

	parser.add_argument("--plot-freq", dest="plot_freq", type=int, default=100, 
		help="Frequency for plotting (default is 100).")

	parser.add_argument("-d", "--dim", dest="embedding_dim", type=int,
		help="Dimension of embeddings for each layer (default is 2).", default=2)

	parser.add_argument("-p", dest="p", type=float, default=1.,
		help="node2vec return parameter (default is 1.).")
	parser.add_argument("-q", dest="q", type=float, default=1.,
		help="node2vec in-out parameter (default is 1.).")
	parser.add_argument('--num-walks', dest="num_walks", type=int, default=25, 
		help="Number of walks per source (default is 25).")
	parser.add_argument('--walk-length', dest="walk_length", type=int, default=15, 
		help="Length of random walk from source (default is 15).")

	parser.add_argument("--alpha", dest="alpha", type=float, default=0,
		help="weighting of attributes (default is 0).")


	# parser.add_argument("--second-order", action="store_true", 
	#   help="Use this flag to use second order topological similarity information.")
	parser.add_argument("--no-attributes", action="store_true", 
		help="Use this flag to not use attributes.")
	# parser.add_argument("--add-attributes", action="store_true", 
	#   help="Use this flag to add attribute sim to adj.")
	parser.add_argument("--multiply-attributes", action="store_true", 
		help="Use this flag to multiply attribute sim to adj.")
	parser.add_argument("--jump-prob", dest="jump_prob", type=float, default=0, 
		help="Probability of randomly jumping to a similar node when walking.")

	parser.add_argument("-v", "--verbose", dest="verbose", action="store_true", 
		help="Use this flag to set verbosity of training.")
	parser.add_argument('--workers', dest="workers", type=int, default=2, 
		help="Number of worker threads to generate training patterns (default is 2).")

	# parser.add_argument("--distance", dest="distance", action="store_true", 
	#   help="Use this flag to use hyperbolic distance loss.")
	parser.add_argument("--sigmoid", dest="sigmoid", action="store_true", 
		help="Use this flag to use sigmoid loss.")
	parser.add_argument("--softmax", dest="softmax", action="store_true", 
		help="Use this flag to use softmax loss.")
	parser.add_argument("--euclidean", dest="euclidean", action="store_true", 
		help="Use this flag to use euclidean negative sampling loss.")

	
	
	parser.add_argument("--plot", dest="plot_path", default="plots/", 
		help="path to save plots (default is 'plots/)'.")
	# parser.add_argument("--embeddings", dest="embedding_path", default="../embeddings/", 
	#   help="path to save embeddings (default is '../embeddings/)'.")
	parser.add_argument("--logs", dest="log_path", default="logs/", 
		help="path to save logs (default is 'logs/)'.")
	# parser.add_argument("--boards", dest="board_path", default="../tensorboards/", 
	#   help="path to save tensorboards (default is '../tensorboards/)'.")
	parser.add_argument("--walks", dest="walk_path", default="walks/", 
		help="path to save random walks (default is 'walks/)'.")
	parser.add_argument("--model", dest="model_path", default="models/", 
		help="path to save model after each epoch (default is 'models/)'.")
	parser.add_argument("--test-results", dest="test_results_path", default="test_results/", 
		help="path to save test results (default is 'test_results/)'.")

	parser.add_argument('--no-gpu', action="store_true", help='flag to train on cpu')

	parser.add_argument('--only-lcc', action="store_true", help='flag to train on only lcc')

	parser.add_argument('--evaluate-class-prediction', action="store_true", help='flag to evaluate class prediction')
	parser.add_argument('--evaluate-link-prediction', action="store_true", help='flag to evaluate link prediction')

	parser.add_argument('--just-walks', action="store_true", help='flag to generate walks with given parameters')

	parser.add_argument('--no-load', action="store_true", help='flag to terminate program if trained model exists')

	parser.add_argument('--directed', action="store_true", help='flag to train on directed graph')


	parser.add_argument('--use-generator', action="store_true", help='flag to train using a generator')

	parser.add_argument('--no-non-edges', action="store_true", help='flag to not add non edges to training graph')
	parser.add_argument('--add-non-edges', action="store_true", help='flag to add non edges to training graph')


	args = parser.parse_args()
	return args

def touch(path):
    with open(path, 'a'):
        os.utime(path, None)

def configure_paths(args):
	'''
	build directories on local system for output of model after each epoch
	'''

	dataset = args.dataset
	if args.directed:
		dataset += "_directed"
	
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


	# if args.second_order:
	#   directory += "second_order_sim/"


	args.plot_path = os.path.join(args.plot_path, dataset)
	if not os.path.exists(args.plot_path):
		os.makedirs(args.plot_path)
	args.plot_path = os.path.join(args.plot_path, directory)
	if not os.path.exists(args.plot_path):
		os.makedirs(args.plot_path)

	# args.embedding_path = os.path.join(args.embedding_path, dataset)
	# if not os.path.exists(args.embedding_path):
	#   os.makedirs(args.embedding_path)
	# args.embedding_path = os.path.join(args.embedding_path, directory)
	# if not os.path.exists(args.embedding_path):
	#   os.makedirs(args.embedding_path)

	args.log_path = os.path.join(args.log_path, dataset)
	if not os.path.exists(args.log_path):
		os.makedirs(args.log_path)
	args.log_path = os.path.join(args.log_path, directory)
	if not os.path.exists(args.log_path):
		os.makedirs(args.log_path)
	args.log_path += "{}.csv".format(dataset)

	# args.board_path = os.path.join(args.board_path, dataset)
	# if not os.path.exists(args.board_path):
	#   os.makedirs(args.board_path)
	# args.board_path = os.path.join(args.board_path, directory)
	# if not os.path.exists(args.board_path):
	#   os.makedirs(args.board_path)

	args.walk_path = os.path.join(args.walk_path, dataset)
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)
	args.walk_path += "/seed={}/".format(args.seed)
	if args.only_lcc:
		args.walk_path += "lcc/"
	else:
		args.walk_path += "all_components/"
	if args.evaluate_link_prediction:
		args.walk_path += "eval_lp/"
		if args.add_non_edges:
			args.walk_path += "add_non_edges/"
		else:
			args.walk_path += "no_non_edges/"
	else:
		args.walk_path += "no_lp/"
	if not os.path.exists(args.walk_path):
		os.makedirs(args.walk_path)

	args.model_path = os.path.join(args.model_path, dataset)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)
	args.model_path = os.path.join(args.model_path, directory)
	if not os.path.exists(args.model_path):
		os.makedirs(args.model_path)

	args.test_results_path = os.path.join(args.test_results_path, dataset)
	if not os.path.exists(args.test_results_path):
		os.makedirs(args.test_results_path)

	# remove seed from directoy path
	s = directory.split("/")
	test_results_directory = "/".join(s[:1] + s[2:])

	args.test_results_path = os.path.join(args.test_results_path, test_results_directory)
	if not os.path.exists(args.test_results_path):
		os.makedirs(args.test_results_path)

	args.test_results_filename = os.path.join(args.test_results_path, "test_results.csv")
	args.test_results_lock_filename = os.path.join(args.test_results_path, "test_results.lock")

	# touch lock file to ensure that it exists
	touch(args.test_results_lock_filename)


def main():

	args = parse_args()
	args.num_positive_samples = 1

	sys.stdout.flush()

	assert not sum([args.multiply_attributes, args.alpha>0, args.jump_prob>0]) > 1

	random.seed(args.seed)
	np.random.seed(args.seed)
	tf.set_random_seed(args.seed)

	configure_paths(args)

	train = True
	if args.no_load:
		plots = os.listdir(args.plot_path)
		if len(plots) > 0 and any(["test.png" in plot for plot in plots]):
			print ("Training already completed and no-load flag is raised -- evaluating")
			train = False

	dataset = args.dataset
	if dataset == "karate":
		topology_graph, features, labels = load_karate(args)
	elif dataset in ["cora", "cora_ml", "pubmed", "citeseer"]:
		# topology_graph, features, labels = load_labelled_attributed_network(dataset, args)
		topology_graph, features, labels, label_info = load_g2g_datasets(dataset, args)
	elif dataset == "ppi":
		topology_graph, features, labels = load_ppi(args)
	elif dataset == "tf_interaction":
		topology_graph, features, labels, label_info = load_tf_interaction(args)
	elif dataset == "wordnet":
		topology_graph, features, labels, label_info = load_wordnet(args)
	else:
		raise Exception


	if not args.evaluate_link_prediction:
		args.evaluate_class_prediction = labels is not None

	# original edges for reconstruction
	reconstruction_edges = topology_graph.edges()
	non_edges = list(nx.non_edges(topology_graph))

	# if args.directed:
	# 	non_edges = list(set(non_edges) - {(v, u) for u, v in reconstruction_edges})

	# if args.verbose:
	print ("determined reconstruction edges and non-edges")


	if features is not None:
		feature_sim = cosine_similarity(features)
		feature_sim -= np.identity(len(features))
		feature_sim [feature_sim  < args.rho] = 0
	else:
		feature_sim = None

	if args.evaluate_link_prediction:
		train_edges, (val_edges, val_non_edges), (test_edges, test_non_edges) = split_edges(reconstruction_edges, non_edges, args)
		n1, e1 = len(topology_graph), len(topology_graph.edges())
		print ("number of validation edges: {}".format(len(val_edges)))
		print ("number of test edges: {}".format(len(test_edges)))
		print ("removing {} edges from training set".format(len(val_edges) + len(test_edges)))
		topology_graph.remove_edges_from(val_edges + test_edges)
		n2, e2 = len(topology_graph), len(topology_graph.edges())
		if args.add_non_edges:
			print ("adding {} non edges as edges to training set".format(len(val_non_edges) + len(test_non_edges)))
			topology_graph.add_edges_from(val_non_edges + test_non_edges)
			nx.set_edge_attributes(topology_graph, "weight", 1)
		n3, e3 = len(topology_graph), len(topology_graph.edges())
		assert n1 == n2 == n3
		assert e1 > e2 
		assert len(val_edges) == len(val_non_edges)
		assert len(test_edges) == len(test_non_edges)

	else:
		train_edges = reconstruction_edges
		val_edges = None
		test_edges = None
		val_non_edges = None
		test_non_edges = None

	if args.alpha > 0:
		walk_file = os.path.join(args.walk_path, "add_attributes_alpha={}".format(args.alpha))
		g = nx.from_numpy_matrix((1 - args.alpha) * nx.adjacency_matrix(topology_graph).A + args.alpha * feature_sim)
	elif args.multiply_attributes:
		walk_file = os.path.join(args.walk_path, "multiply_attributes")
		A = nx.adjacency_matrix(topology_graph).A
		g = nx.from_numpy_matrix(A * feature_sim)
	elif args.jump_prob > 0:
		walk_file = os.path.join(args.walk_path, "jump_prob={}".format(args.jump_prob))
		g = topology_graph
	else:
		walk_file = os.path.join(args.walk_path, "no_attributes")
		g = topology_graph
	walk_file += "_num_walks={}-walk_len={}-p={}-q={}.walk".format(args.num_walks, 
				args.walk_length, args.p, args.q)

	walks = load_walks(g, walk_file, feature_sim, args)

	if args.just_walks:
		return
		
	positive_samples, negative_samples, alias_dict =\
		determine_positive_and_negative_samples(nodes=topology_graph.nodes(), 
		walks=walks, context_size=args.context_size, directed=args.directed)

	num_nodes = len(topology_graph)
	num_steps = int((len(positive_samples) + args.batch_size - 1) / args.batch_size)
	model, initial_epoch = build_model(num_nodes, args)
	optimizer = ("adam" if args.euclidean else
		ExponentialMappingOptimizer(learning_rate=args.lr)
	)
	loss = (
		hyperbolic_softmax_loss(args.directed * 0)
		if args.softmax 
		else hyperbolic_sigmoid_loss
		if args.sigmoid 
		else euclidean_negative_sampling_loss
		if args.euclidean
		else hyperbolic_negative_sampling_loss(r=args.r, t=args.t)
	)
	# optimizer=tf.train.GradientDescentOptimizer(0.05)
	model.compile(optimizer=optimizer, loss=loss)
	model.summary()

	if args.evaluate_link_prediction:
		val_data = make_validation_data(val_edges, negative_samples, args)
	else:
		val_data = None

	# non_edge_dict = convert_edgelist_to_dict(non_edges)
	# print ("determined true non edges")
	# val_in, val_target = make_validation_data(reconstruction_edges, non_edge_dict, args)
	print ("determined validation data")

	if args.evaluate_link_prediction:
		# monitor = "mean_roc_reconstruction"
		# monitor = "map_lp"
		# mode = "max"
		monitor = "val_loss"
		mode = "min"
	elif args.evaluate_class_prediction:
		monitor = "0.1_micro"
		mode = "max"
		# monitor = "val_loss"
		# mode = "min"
	else:
		monitor = "val_loss"#"mean_rank_reconstruction"
		mode = "min"
	early_stopping = EarlyStopping(monitor=monitor, mode=mode, patience=args.patience, verbose=1)
	logger = PeriodicStdoutLogger(reconstruction_edges, non_edges, val_edges, val_non_edges, labels, label_info,
				n=args.plot_freq, epoch=initial_epoch, args=args) 

	callbacks=[
		TerminateOnNaN(), 
		logger,
		CSVLogger(args.log_path, append=True), 
		early_stopping
	]

	if train:

		if args.use_generator:
			print ("Training with data generator")
			random.shuffle(positive_samples)
			training_gen = TrainingSequence(positive_samples, negative_samples, alias_dict, args)

			sys.stdout.flush()

			model.fit_generator(training_gen, 
				workers=args.workers, max_queue_size=25, use_multiprocessing=args.workers>0, steps_per_epoch=num_steps, 
				epochs=args.num_epochs, initial_epoch=initial_epoch, verbose=args.verbose,
				validation_data=val_data,
				callbacks=callbacks
			)

		else:
			print ("Training without data generator")

			x = get_training_sample(np.array(positive_samples), negative_samples, args.num_negative_samples, alias_dict)
			y = np.zeros(list(x.shape) + [1])
			print ("determined training samples")

			sys.stdout.flush()

			model.fit(x, y, batch_size=args.batch_size, 
				epochs=args.num_epochs, initial_epoch=initial_epoch, verbose=args.verbose,
				validation_data=val_data,
				callbacks=callbacks
			)

	print ("Training completed -- loading best model according to {}".format(monitor))

	saved_models = sorted([f for f in os.listdir(args.model_path) 
		if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
	model_file = os.path.join(args.model_path, saved_models[0])

	print ("Determined best model filename: {}".format(model_file))
	embedding = load_embedding(model_file)
	print (embedding)

	print ("removing all other saved models")
	for saved_model in saved_models[1:]:
		old_model_path = os.path.join(self.args.model_path, saved_model)
		print ("removing {}".format(old_model_path))
		os.remove(old_model_path)

	# hyperboloid_embedding = model.layers[-1].get_weights()[0]
	if args.euclidean:
		dists = euclidean_distances(embedding)
	else:
		dists = hyperbolic_distance_hyperboloid_pairwise(embedding, embedding)
	
	print ("Evaluating on test data")

	test_results = dict()

	(mean_rank_reconstruction, map_reconstruction, 
		mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
		reconstruction_edges, non_edges)

	test_results.update({"mean_rank_reconstruction": mean_rank_reconstruction, 
			"map_reconstruction": map_reconstruction,
			"mean_roc_reconstruction": mean_roc_reconstruction})

	if args.evaluate_link_prediction:
		(mean_rank_lp, map_lp, 
		mean_roc_lp) = evaluate_rank_and_MAP(dists, test_edges, test_non_edges)

		test_results.update({"mean_rank_lp": mean_rank_lp, 
				"map_lp": map_lp,
				"mean_roc_lp": mean_roc_lp})

	else:
		mean_rank_lp, map_lp, mean_roc_lp = None, None, None 

	if not args.euclidean:
		poincare_embedding = hyperboloid_to_poincare_ball(embedding)
		klein_embedding = hyperboloid_to_klein(embedding)

	epoch = logger.epoch

	plot_path = os.path.join(args.plot_path, "epoch_{:05d}_plot_test.png".format(epoch) )
	if args.euclidean:
		plot_euclidean_embedding(epoch, reconstruction_edges, 
			embedding,
			labels, label_info,
			mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
			mean_rank_lp, map_lp, mean_roc_lp,
			plot_path)
	else:
		plot_disk_embeddings(epoch, reconstruction_edges, 
			poincare_embedding, 
			labels, label_info,
			mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
			mean_rank_lp, map_lp, mean_roc_lp,
			plot_path)

	roc_path = os.path.join(args.plot_path, "epoch_{:05d}_roc_curve_test.png".format(epoch) )
	plot_roc(dists, reconstruction_edges, non_edges, test_edges, test_non_edges, roc_path)

	precision_recall_path = os.path.join(args.plot_path, 
		"epoch_{:05d}_precision_recall_curve_test.png".format(epoch) )
	plot_precisions_recalls(dists, reconstruction_edges, non_edges, test_edges, test_non_edges, precision_recall_path)

	if args.evaluate_class_prediction:
		if args.euclidean:
			label_percentages, f1_micros, f1_macros = evaluate_classification(embedding, labels, args)
		else:
			label_percentages, f1_micros, f1_macros = evaluate_classification(klein_embedding, labels, args)

		print (f1_micros)

		for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
				test_results.update({"{}_micro".format(label_percentage): f1_micro})
				test_results.update({"{}_macro".format(label_percentage): f1_macro})

		f1_path = os.path.join(args.plot_path, "epoch_{:05d}_class_prediction_f1_test.png".format(epoch))
		plot_classification(label_percentages, f1_micros, f1_macros, f1_path)

	print ("saving test results to: {}".format(args.test_results_filename))
	threadsafe_save_test_results(args.test_results_lock_filename, args.test_results_filename, 
		args.seed, test_results)

if __name__ == "__main__":
	main()