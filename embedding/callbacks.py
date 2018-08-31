from __future__ import print_function

import re
import sys
import os
import numpy as np

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve

from keras.callbacks import Callback
from utils import convert_edgelist_to_dict

from metrics import evaluate_rank_and_MAP, evaluate_classification


def minkowski_dot_np(x, y):
	assert len(x.shape) == 2
	rank = x.shape[1] - 1
	return np.sum(x[:,:rank] * y[:,:rank], axis=-1, keepdims=True) - x[:,rank:] * y[:,rank:]

def minkowski_dot_pairwise(u, v):
	"""
	`u` and `v` are vectors in Minkowski space.
	"""
	rank = u.shape[-1] - 1
	euc_dp = u[:,:rank].dot(v[:,:rank].T)
	return euc_dp - u[:,rank, None] * v[:,rank]

def hyperbolic_distance_hyperboloid_pairwise(X, Y):
	inner_product = minkowski_dot_pairwise(X, Y)
	inner_product = np.minimum(inner_product, -(1 + 1e-32))
	# inner_product = np.clip(inner_product, a_max=-1, a_min=-np.inf)
	return np.arccosh(-inner_product)

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def plot_euclidean_embedding(epoch, edges, euclidean_embedding, labels, label_info,
	mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
	mean_rank_lp, map_lp, mean_roc_lp, path):

	# if len(labels.shape) > 1:
		# raise Exception	
		# unique_labels = np.unique(labels, axis=0)
		# labels = np.array([np.where((unique_labels == label).all(axis=-1))[0][0] for label in labels])

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges)

	if labels is not None:
		num_classes = len(set(labels))
		colors = np.random.rand(num_classes, 3)

	print ("saving plot to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Epoch={:05d}, Mean_rank_recon={}, MAP_recon={}, Mean_AUC_recon={}".format(epoch, 
		mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction)
	if mean_rank_lp is not None:
		title += "\nMean_rank_lp={}, MAP_lp={}, Mean_AUC_lp={}".format(mean_rank_lp,
			map_lp, mean_roc_lp)
	plt.suptitle(title)
	
	plt.title("Euclidean")
	u_emb = euclidean_embedding[edges[:,0]]
	v_emb = euclidean_embedding[edges[:,1]]
	# for u, v in edges:
	# 	u_emb = poincare_embedding[u]
	# 	v_emb = poincare_embedding[v]
	# 	plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)

	if labels is None:
		plt.scatter(euclidean_embedding[:,0], euclidean_embedding[:,1], s=10, c="r", zorder=1)

	else:

		for c in range(num_classes):
			idx = labels == c
			plt.scatter(euclidean_embedding[idx,0], euclidean_embedding[idx,1], s=10, c=colors[c], 
				label=label_info[c] if label_info is not None else None, zorder=1)
		
	plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=3)
	plt.savefig(path)
	plt.close()


def plot_disk_embeddings(epoch, edges, poincare_embedding, labels, label_info,
	mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
	mean_rank_lp, map_lp, mean_roc_lp, path):

	# def bit_shift(bitlist):
	#     out = 0
	#     for bit in bitlist:
	#         out = (out << 1) | bit
	#     return out


	# if len(labels.shape) > 1:
	# 	raise Exception
	# 	unique_labels = np.unique(labels, axis=0)
	# 	labels = np.array([np.where((unique_labels == label).all(axis=-1))[0][0] for label in labels])

	if not isinstance(edges, np.ndarray):
		edges = np.array(edges)

	if labels is not None:
		num_classes = len(set(labels))
		colors = np.random.rand(num_classes, 3)

	# num_classes = len(set(labels))
	# colors = np.random.rand(num_classes, 3)

	print ("saving plot to {}".format(path))

	fig = plt.figure(figsize=[14, 7])
	title = "Epoch={:05d}, Mean_rank_recon={}, MAP_recon={}, Mean_AUC_recon={}".format(epoch, 
		mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction)
	if mean_rank_lp is not None:
		title += "\nMean_rank_lp={}, MAP_lp={}, Mean_AUC_lp={}".format(mean_rank_lp,
			map_lp, mean_roc_lp)
	plt.suptitle(title)
	
	ax = fig.add_subplot(111)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	u_emb = poincare_embedding[edges[:,0]]
	v_emb = poincare_embedding[edges[:,1]]
	# for u, v in edges:
	# 	u_emb = poincare_embedding[u]
	# 	v_emb = poincare_embedding[v]
	# 	plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)
	if labels is None:
		plt.scatter(poincare_embedding[:,0], poincare_embedding[:,1], s=10, c="r", zorder=1)
	else:
		for c in range(num_classes):
			idx = labels == c
			plt.scatter(poincare_embedding[idx,0], poincare_embedding[idx,1], s=10, c=colors[c], 
				label=label_info[c] if label_info is not None else None, zorder=1)
	# plt.scatter(poincare_embedding[:,0], poincare_embedding[:,1], s=10, c=colors[labels], zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	# ax = fig.add_subplot(122)
	# plt.title("Klein")
	# ax.add_artist(plt.Circle([0,0], 1, fill=False))
	# u_emb = klein_embedding[edges[:,0]]
	# v_emb = klein_embedding[edges[:,1]]
	# # for u, v in edges:
	# # 	u_emb = klein_embedding[u]
	# # 	v_emb = klein_embedding[v]
	# # 	plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	# plt.plot([u_emb[:,0], v_emb[:,0]], [u_emb[:,1], v_emb[:,1]], c="k", linewidth=0.05, zorder=0)
	# for c in range(num_classes):
	# 	idx = labels == c
	# 	plt.scatter(klein_embedding[idx,0], klein_embedding[idx,1], s=10, c=colors[c], 
	# 		label=label_info[c] if label_info is not None else None, zorder=1)
	# # plt.scatter(klein_embedding[:,0], klein_embedding[:,1], s=10, c=c[labels], zorder=1)
	# plt.xlim([-1,1])
	# plt.ylim([-1,1])

	# Shrink current axis by 20%
	box = ax.get_position()
	ax.set_position([box.x0, box.y0, box.width * 0.5, box.height])

	# Put a legend to the right of the current axis
	ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), ncol=1)

	plt.savefig(path)
	plt.close()

def plot_precisions_recalls(dists, reconstruction_edges, non_edges, val_edges, val_non_edges, path):

	print ("saving precision recall curves to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Embedding quality precision-recall curve"
	plt.suptitle(title)

	reconstruction_edges = np.array(reconstruction_edges)
	non_edges = np.array(non_edges) 

	edge_dists = dists[reconstruction_edges[:,0], reconstruction_edges[:,1]]
	non_edge_dists = dists[non_edges[:,0], non_edges[:,1]]

	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	_dists = np.append(edge_dists, non_edge_dists)

	precisions, recalls, _ = precision_recall_curve(targets, -_dists)

	plt.plot(recalls, precisions, c="r")

	legend = ["reconstruction"]

	if val_edges is not None:
		val_edges = np.array(val_edges)
		val_edge_dists = dists[val_edges[:,0], val_edges[:,1]]

		val_non_edges = np.array(val_non_edges)
		val_non_edge_dists = dists[val_non_edges[:,0], val_non_edges[:,1]]

		targets = np.append(np.ones_like(val_edge_dists), np.zeros_like(val_non_edge_dists))
		_dists = np.append(val_edge_dists, val_non_edge_dists)

		precisions, recalls, _ = precision_recall_curve(targets, -_dists)

		plt.plot(recalls, precisions, c="b")

		legend += ["link prediction"]


	plt.xlabel("recall")
	plt.ylabel("precision")
	plt.legend(legend)
	plt.savefig(path)
	plt.close()


def plot_roc(dists, reconstruction_edges, non_edges, val_edges, val_non_edges, path):

	print ("saving roc plot to {}".format(path))

	fig = plt.figure(figsize=[7, 7])
	title = "Embedding quality ROC curve"
	plt.suptitle(title)

	reconstruction_edges = np.array(reconstruction_edges)
	non_edges = np.array(non_edges) 

	edge_dists = dists[reconstruction_edges[:,0], reconstruction_edges[:,1]]
	non_edge_dists = dists[non_edges[:,0], non_edges[:,1]]

	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	_dists = np.append(edge_dists, non_edge_dists)

	fpr, tpr, _ = roc_curve(targets, -_dists)
	auc = roc_auc_score(targets, -_dists)
	precisions, recalls, _ = precision_recall_curve(targets, -_dists)

	plt.plot(fpr, tpr, c="r")

	legend = ["reconstruction AUC={}".format(auc)]

	if val_edges is not None:
		val_edges = np.array(val_edges)
		val_edge_dists = dists[val_edges[:,0], val_edges[:,1]]

		val_non_edges = np.array(val_non_edges)
		val_non_edge_dists = dists[val_non_edges[:,0], val_non_edges[:,1]]

		targets = np.append(np.ones_like(val_edge_dists), np.zeros_like(val_non_edge_dists))
		_dists = np.append(val_edge_dists, val_non_edge_dists)

		fpr, tpr, _ = roc_curve(targets, -_dists)
		auc = roc_auc_score(targets, -_dists)

		plt.plot(fpr, tpr, c="b")

		legend += ["link prediction AUC={}".format(auc)]

	plt.plot([0,1], [0,1], c="k")

	plt.xlabel("fpr")
	plt.ylabel("tpr")
	plt.legend(legend)
	plt.savefig(path)
	plt.close()

def plot_classification(label_percentages, f1_micros, f1_macros, path):

	print ("saving classification plot to {}".format(path))


	fig = plt.figure(figsize=[7, 7])
	title = "Node classification"
	plt.suptitle(title)
	
	plt.plot(label_percentages, f1_micros, c="r")
	plt.plot(label_percentages, f1_macros, c="b")
	plt.legend(["f1_micros", "f1_macros"])
	plt.xlabel("label_percentages")
	plt.ylabel("f1 score")
	plt.ylim([0,1])
	plt.savefig(path)
	plt.close()

class PeriodicStdoutLogger(Callback):

	def __init__(self, reconstruction_edges, non_edges, val_edges, val_non_edges, labels, label_info,
		epoch, n, args):
		self.reconstruction_edges = reconstruction_edges
		self.non_edges = non_edges

		# self.reconstruction_edge_dict = convert_edgelist_to_dict(reconstruction_edges)
		self.val_edges = val_edges
		self.val_non_edges = val_non_edges
		# self.val_edge_dict = convert_edgelist_to_dict(val_edges)
		# self.non_edge_dict = convert_edgelist_to_dict(non_edges)
		# self.non_edge_dict = non_edge_dict
		self.labels = labels
		self.label_info = label_info
		self.epoch = epoch
		self.n = n
		self.args = args

	def on_epoch_end(self, batch, logs={}):
	
		self.epoch += 1

		s = "Completed epoch {}, loss={}".format(self.epoch, logs["loss"])
		if "val_loss" in logs.keys():
			s += ", val_loss={}".format(logs["val_loss"])
		print (s)

		hyperboloid_embedding = self.model.layers[-1].get_weights()[0]
		# print (hyperboloid_embedding)
		# print (minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding))

		if self.args.euclidean:
			dists = euclidean_distances(hyperboloid_embedding)
		else:
			dists = hyperbolic_distance_hyperboloid_pairwise(hyperboloid_embedding, hyperboloid_embedding)

		# print minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding)

		# if self.args.verbose:
		print ("reconstruction")
		# (mean_rank_reconstruction, map_reconstruction, 
		# 	mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
		# 	self.reconstruction_edge_dict, self.non_edge_dict)
		(mean_rank_reconstruction, map_reconstruction, 
			mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
			self.reconstruction_edges, self.non_edges)

		logs.update({"mean_rank_reconstruction": mean_rank_reconstruction, 
			"map_reconstruction": map_reconstruction,
			"mean_roc_reconstruction": mean_roc_reconstruction})


		if self.args.evaluate_link_prediction:
			# if self.args.verbose:
			print ("link prediction")
			# (mean_rank_lp, map_lp, 
			# mean_roc_lp) = evaluate_rank_and_MAP(dists, 
			# self.val_edge_dict, self.non_edge_dict)

			(mean_rank_lp, map_lp, 
			mean_roc_lp) = evaluate_rank_and_MAP(dists, 
			self.val_edges, self.val_non_edges)

			logs.update({"mean_rank_lp": mean_rank_lp, 
				"map_lp": map_lp,
				"mean_roc_lp": mean_roc_lp})
		else:

			mean_rank_lp, map_lp, mean_roc_lp = None, None, None

		if self.args.euclidean:
			poincare_embedding = hyperboloid_embedding
			klein_embedding = hyperboloid_embedding
		else:
			poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
			klein_embedding = hyperboloid_to_klein(hyperboloid_embedding)

		if self.args.evaluate_class_prediction:
			label_percentages, f1_micros, f1_macros = evaluate_classification(klein_embedding, self.labels, self.args)

			print (f1_micros)

			for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
				logs.update({"{}_micro".format(label_percentage): f1_micro})
				logs.update({"{}_macro".format(label_percentage): f1_macro})



		if self.epoch % self.n == 0:

			# if self.args.embedding_dim == 2:
			plot_path = os.path.join(self.args.plot_path, "epoch_{:05d}_plot.png".format(self.epoch) )
			if self.args.euclidean:
				plot_euclidean_embedding(self.epoch, self.reconstruction_edges, 
					poincare_embedding,
					self.labels, self.label_info,
					mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
					mean_rank_lp, map_lp, mean_roc_lp,
					plot_path)

			else:
				plot_disk_embeddings(self.epoch, self.reconstruction_edges, 
					poincare_embedding,
					self.labels, self.label_info,
					mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
					mean_rank_lp, map_lp, mean_roc_lp,
					plot_path)
			# else:
			# 	print ("dim > 2, omitting plot")

			roc_path = os.path.join(self.args.plot_path, "epoch_{:05d}_roc_curve.png".format(self.epoch) )
			plot_roc(dists, self.reconstruction_edges, self.non_edges, 
				self.val_edges, self.val_non_edges, roc_path)

			precision_recall_path = os.path.join(self.args.plot_path, "epoch_{:05d}_precision_recall_curve.png".format(self.epoch) )
			plot_precisions_recalls(dists, self.reconstruction_edges, self.non_edges, 
				self.val_edges, self.val_non_edges, precision_recall_path)

			if self.args.evaluate_class_prediction:
				f1_path = os.path.join(self.args.plot_path, "epoch_{:05d}_class_prediction_f1.png".format(self.epoch))
				plot_classification(label_percentages, f1_micros, f1_macros, f1_path)

		self.remove_old_models()
		self.save_model()

		sys.stdout.flush()

	def remove_old_models(self):
		old_models = sorted([f for f in os.listdir(self.args.model_path) 
			if re.match(r"^[0-9][0-9][0-9][0-9]*", f)])
		for model in old_models:
			old_model_path = os.path.join(self.args.model_path, model)
			print ("removing model: {}".format(old_model_path))
			os.remove(old_model_path)


	def save_model(self):
		filename = os.path.join(self.args.model_path, "{:05d}.h5".format(self.epoch))
		print("saving model to {}".format(filename))
		self.model.save_weights(filename)