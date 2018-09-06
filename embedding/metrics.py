from __future__ import print_function

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, roc_curve, precision_recall_curve
from sklearn.linear_model import LogisticRegressionCV
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import StratifiedShuffleSplit

def evaluate_rank_and_MAP(dists, edgelist, non_edgelist):
	assert not isinstance(edgelist, dict)

	if not isinstance(edgelist, np.ndarray):
		edgelist = np.array(edgelist)

	if not isinstance(non_edgelist, np.ndarray):
		non_edgelist = np.array(non_edgelist)

	edge_dists = dists[edgelist[:,0], edgelist[:,1]]
	non_edge_dists = dists[non_edgelist[:,0], non_edgelist[:,1]]

	targets = np.append(np.ones_like(edge_dists), np.zeros_like(non_edge_dists))
	ap_score = average_precision_score(targets, -np.append(edge_dists, non_edge_dists))
	auc_score = roc_auc_score(targets, -np.append(edge_dists, non_edge_dists))


	idx = non_edge_dists.argsort()
	ranks = np.searchsorted(non_edge_dists, edge_dists, sorter=idx).mean()

	print ("MEAN RANK =", ranks, "AP =", ap_score, 
		"ROC AUC =", auc_score)

	return ranks, ap_score, auc_score

# def evaluate_rank_and_MAP(dists, edge_dict, non_edge_dict):

# 	ranks = []
# 	ap_scores = []
# 	roc_auc_scores = []
	
# 	for u, neighbours in edge_dict.items():
# 		_dists = dists[u, neighbours + non_edge_dict[u]]
# 		_labels = np.append(np.ones(len(neighbours)), np.zeros(len(non_edge_dict[u])))
# 		# _dists = dists[u]
# 		# _dists[u] = 1e+12
# 		# _labels = np.zeros(embedding.shape[0])
# 		# _dists_masked = _dists.copy()
# 		# _ranks = []
# 		# for v in v_set:
# 		# 	_labels[v] = 1
# 		# 	_dists_masked[v] = np.Inf
# 		ap_scores.append(average_precision_score(_labels, -_dists))
# 		roc_auc_scores.append(roc_auc_score(_labels, -_dists))

# 		neighbour_dists = dists[u, neighbours]
# 		non_neighbour_dists = dists[u, non_edge_dict[u]]
# 		idx = non_neighbour_dists.argsort()
# 		_ranks = np.searchsorted(non_neighbour_dists, neighbour_dists, sorter=idx) + 1

# 		# _ranks = []
# 		# _dists_masked = _dists.copy()
# 		# _dists_masked[:len(neighbours)] = np.inf

# 		# for v in neighbours:
# 		# 	d = _dists_masked.copy()
# 		# 	d[v] = _dists[v]
# 		# 	r = np.argsort(d)
# 		# 	raise Exception
# 		# 	_ranks.append(np.where(r==v)[0][0] + 1)

# 		ranks.append(np.mean(_ranks))
# 	print ("MEAN RANK=", np.mean(ranks), "MEAN AP=", np.mean(ap_scores), 
# 		"MEAN ROC AUC=", np.mean(roc_auc_scores))
# 	return np.mean(ranks), np.mean(ap_scores), np.mean(roc_auc_scores)

def evaluate_classification(klein_embedding, labels, args,
	label_percentages=np.arange(0.02, 0.11, 0.01), n_repeats=10):

	# def idx_shuffle(labels):
	# 	class_memberships = [list(np.random.permutation(np.where(labels==c)[0])) for c in sorted(set(labels))]
	# 	idx = []
	# 	while len(class_memberships) > 0:
	# 		for _class in class_memberships:
	# 			idx.append(_class.pop(0))
	# 			if len(_class) == 0:
	# 				class_memberships.remove(_class)
		# return idx

	# np.random.seed(args.seed)

	assert len(labels.shape) == 1

	num_nodes, dim = klein_embedding.shape

	f1_micros = np.zeros((n_repeats, len(label_percentages)))
	f1_macros = np.zeros((n_repeats, len(label_percentages)))

	# if len(labels.shape) > 1:
	# 	classes = np.arange(labels.shape[1])
	# 	idx = np.random.permutation(num_nodes)
	# else:
	# 	classes = sorted(set(labels))
	# 	idx = idx_shuffle(labels)
	
	model = LogisticRegressionCV()

	for seed in range(n_repeats):
	
		for i, label_percentage in enumerate(label_percentages):

			sss = StratifiedShuffleSplit(n_splits=1, test_size=1-label_percentage, random_state=seed)
			split_train, split_test = next(sss.split(klein_embedding, labels))
			# num_labels = int(max(num_nodes * label_percentage, len(classes)))
			# idx = np.random.permutation(num_nodes)
			# if len(labels.shape) > 1:
			# 	model =  OneVsRestClassifier(LogisticRegression(random_state=0))
			model.fit(klein_embedding[split_train], labels[split_train])
			predictions = model.predict(klein_embedding[split_test])
			f1_micro = f1_score(labels[split_test], predictions, average="micro")
			f1_macro = f1_score(labels[split_test], predictions, average="macro")
			f1_micros[seed,i] = f1_micro
			f1_macros[seed,i] = f1_macro

	return label_percentages, f1_micros.mean(axis=0), f1_macros.mean(axis=0)




# def evaluate_lexical_entailment(embedding):

# 	def is_a_score(u, v, alpha=1e3):
# 		return -(1 + alpha * (np.linalg.norm(v, axis=-1) - np.linalg.norm(u, axis=-1))) * hyperbolic_distance(u, v)

# 	print ("evaluating lexical entailment")

# 	hyperlex_noun_idx_df = pd.read_csv("../data/wordnet/hyperlex_idx_ranks.txt", index_col=0, sep=" ")

# 	U = np.array(hyperlex_noun_idx_df["WORD1"], dtype=int)
# 	V = np.array(hyperlex_noun_idx_df["WORD2"], dtype=int)

# 	true_is_a_score = np.array(hyperlex_noun_idx_df["AVG_SCORE_0_10"])
# 	predicted_is_a_score = is_a_score(embedding[U], embedding[V])

# 	r, p = spearmanr(true_is_a_score, predicted_is_a_score)

# 	print (r, p)

# 	return r, p

