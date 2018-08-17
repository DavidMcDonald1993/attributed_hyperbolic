import os
import numpy as np

# import matplotlib
# matplotlib.use('agg')
import matplotlib.pyplot as plt

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
	inner_product = np.clip(inner_product, a_max=-1, a_min=-np.inf)
	return np.arccosh(-inner_product)

def hyperboloid_to_poincare_ball(X):
	return X[:,:-1] / (1 + X[:,-1,None])

def hyperboloid_to_klein(X):
	return X[:,:-1] / X[:,-1,None]

def plot_disk_embeddings(epoch, edges, poincare_embedding, klein_embedding, labels, 
	mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
	mean_rank_lp, map_lp, mean_roc_lp, path):

	if len(labels.shape ) > 1:
		labels = labels[:,0]

	print ("saving plot to {}".format(path))

	fig = plt.figure(figsize=[14, 7])
	title = "Epoch={:05d}, Mean_rank_recon={}, MAP_recon={}, Mean_AUC_recon={}".format(epoch, 
		mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction)
	if mean_rank_lp is not None:
		title += "\nMean_rank_lp={}, MAP_lp={}, Mean_AUC_lp={}".format(mean_rank_lp,
			map_lp, mean_roc_lp)
	plt.suptitle(title)
	
	ax = fig.add_subplot(121)
	plt.title("Poincare")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	for u, v in edges:
		u_emb = poincare_embedding[u]
		v_emb = poincare_embedding[v]
		plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.scatter(poincare_embedding[:,0], poincare_embedding[:,1], s=10, c=labels, zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	ax = fig.add_subplot(122)
	plt.title("Klein")
	ax.add_artist(plt.Circle([0,0], 1, fill=False))
	for u, v in edges:
		u_emb = klein_embedding[u]
		v_emb = klein_embedding[v]
		plt.plot([u_emb[0], v_emb[0]], [u_emb[1], v_emb[1]], c="k", linewidth=0.05, zorder=0)
	plt.scatter(klein_embedding[:,0], klein_embedding[:,1], s=10, c=labels, zorder=1)
	plt.xlim([-1,1])
	plt.ylim([-1,1])

	
	plt.savefig(path)
	plt.close()

def plot_precisions_recalls(dists, reconstruction_edges, removed_edges, non_edges, path):

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

	if removed_edges is not None:
		removed_edges = np.array(removed_edges)
		removed_edge_dists = dists[removed_edges[:,0], removed_edges[:,1]]

		targets = np.append(np.ones_like(removed_edge_dists), np.zeros_like(non_edge_dists))
		_dists = np.append(removed_edge_dists, non_edge_dists)

		precisions, recalls, _ = precision_recall_curve(targets, -_dists)

		plt.plot(recalls, precisions, c="b")

		legend += ["link prediction"]


	plt.xlabel("recall")
	plt.ylabel("precision")
	plt.legend(legend)
	plt.savefig(path)
	plt.close()


def plot_roc(dists, reconstruction_edges, removed_edges, non_edges, path):

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

	if removed_edges is not None:
		removed_edges = np.array(removed_edges)
		removed_edge_dists = dists[removed_edges[:,0], removed_edges[:,1]]

		targets = np.append(np.ones_like(removed_edge_dists), np.zeros_like(non_edge_dists))
		_dists = np.append(removed_edge_dists, non_edge_dists)

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

	def __init__(self, reconstruction_edges, val_edges, non_edges, non_edge_dict, labels, 
		epoch, n, args):
		self.reconstruction_edges = reconstruction_edges
		self.reconstruction_edge_dict = convert_edgelist_to_dict(reconstruction_edges)
		self.val_edges = val_edges
		self.val_edge_dict = convert_edgelist_to_dict(val_edges)
		self.non_edges = non_edges
		# self.non_edge_dict = convert_edgelist_to_dict(non_edges)
		self.non_edge_dict = non_edge_dict
		self.labels = labels
		self.epoch = epoch
		self.n = n
		self.args = args

	def on_epoch_end(self, batch, logs={}):
	
		self.epoch += 1


		if self.args.verbose:
			s = "Completed epoch {}, loss={}".format(self.epoch, logs["loss"])
			if "val_loss" in logs.keys():
				s += ", val_loss={}".format(logs["val_loss"])
			print (s)

		hyperboloid_embedding = self.model.layers[-1].get_weights()[0]
		# print (hyperboloid_embedding)
		# print (minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding))

		dists = hyperbolic_distance_hyperboloid_pairwise(hyperboloid_embedding, hyperboloid_embedding)

		# print minkowski_dot_np(hyperboloid_embedding, hyperboloid_embedding)

		if self.args.verbose:
			print ("reconstruction")
		(mean_rank_reconstruction, map_reconstruction, 
			mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
			self.reconstruction_edge_dict, self.non_edge_dict)
		# (mean_rank_reconstruction, map_reconstruction, 
		# 	mean_roc_reconstruction) = evaluate_rank_and_MAP(dists, 
		# 	self.reconstruction_edges, self.non_edges)

		logs.update({"mean_rank_reconstruction": mean_rank_reconstruction, 
			"map_reconstruction": map_reconstruction,
			"mean_roc_reconstruction": mean_roc_reconstruction})


		if self.args.evaluate_link_prediction:
			if self.args.verbose:
				print ("link prediction")
			(mean_rank_lp, map_lp, 
			mean_roc_lp) = evaluate_rank_and_MAP(dists, 
			self.val_edge_dict, self.non_edge_dict)

			# (mean_rank_lp, map_lp, 
			# mean_roc_lp) = evaluate_rank_and_MAP(dists, 
			# self.val_edges, self.non_edges)

			logs.update({"mean_rank_lp": mean_rank_lp, 
				"map_lp": map_lp,
				"mean_roc_lp": mean_roc_lp})
		else:

			mean_rank_lp, map_lp, mean_roc_lp = None, None, None

		poincare_embedding = hyperboloid_to_poincare_ball(hyperboloid_embedding)
		klein_embedding = hyperboloid_to_klein(hyperboloid_embedding)

		if self.args.evaluate_class_prediction:
			label_percentages, f1_micros, f1_macros = evaluate_classification(klein_embedding, self.labels, self.args)

			print (f1_micros)

			for label_percentage, f1_micro, f1_macro in zip(label_percentages, f1_micros, f1_macros):
				logs.update({"{}_micro".format(label_percentage): f1_micro})
				logs.update({"{}_macro".format(label_percentage): f1_macro})



		if self.epoch % self.n == 0:

			plot_path = os.path.join(self.args.plot_path, "epoch_{:05d}_plot.png".format(self.epoch) )
			plot_disk_embeddings(self.epoch, self.reconstruction_edges, 
				poincare_embedding, klein_embedding,
				self.labels, 
				mean_rank_reconstruction, map_reconstruction, mean_roc_reconstruction,
				mean_rank_lp, map_lp, mean_roc_lp,
				plot_path)

			roc_path = os.path.join(self.args.plot_path, "epoch_{:05d}_roc_curve.png".format(self.epoch) )
			plot_roc(dists, self.reconstruction_edges, self.val_edges, self.non_edges, roc_path)

			precision_recall_path = os.path.join(self.args.plot_path, "epoch_{:05d}_precision_recall_curve.png".format(self.epoch) )
			plot_precisions_recalls(dists, self.reconstruction_edges, 
				self.val_edges, self.non_edges, precision_recall_path)

			if self.args.evaluate_class_prediction:
				f1_path = os.path.join(self.args.plot_path, "epoch_{:05d}_class_prediction_f1.png".format(self.epoch))
				plot_classification(label_percentages, f1_micros, f1_macros, f1_path)