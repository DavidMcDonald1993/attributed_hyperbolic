from __future__ import print_function

import random
import numpy as np
import scipy as sp

from utils import get_training_sample

from keras.utils import Sequence

class TrainingSequence(Sequence):

	def __init__(self, positive_samples, negative_samples, alias_dict, args):
		assert isinstance(positive_samples, list)
		self.positive_samples = positive_samples
		self.negative_samples = negative_samples
		self.alias_dict = alias_dict
		self.batch_size = args.batch_size
		self.num_negative_samples = args.num_negative_samples
		# self.binarizer = np.identity(len(alias_dict))

	def alias_draw(self, J, q, size=1):
	    '''
	    Draw sample from a non-uniform discrete distribution using alias sampling.
	    '''
	    K = len(J)
	    kk = np.floor(np.random.uniform(high=K, size=size)).astype(np.int)
	    r = np.random.uniform(size=size)
	    idx = r >= q[kk]
	    kk[idx] = J[kk[idx]]
	    return kk


	def get_training_sample(self, batch_positive_samples):

		negative_samples = self.negative_samples
		num_negative_samples = self.num_negative_samples
		alias_dict = self.alias_dict

		input_nodes = batch_positive_samples[:,0]

		batch_negative_samples = np.array([
			negative_samples[u][self.alias_draw(alias_dict[u][0], alias_dict[u][1], size=num_negative_samples)]
			for u in input_nodes
		], dtype=np.int64)

		batch_nodes = np.append(batch_positive_samples, batch_negative_samples, axis=1)
		return batch_nodes


	def __len__(self):
		return int(np.ceil(len(self.positive_samples) / float(self.batch_size)))

	def __getitem__(self, batch_idx):
		# print ("get batch {}, thread: {}".format(batch_idx, threading.current_thread()))
		batch_size = self.batch_size
		positive_samples = self.positive_samples
		batch_positive_samples = np.array(
			positive_samples[batch_idx * batch_size : (batch_idx + 1) * batch_size], dtype=np.int64)
		training_sample = self.get_training_sample(batch_positive_samples, )
		# training_sample = self.binarizer[training_sample]
		# print training_sample.shape
		# target = np.zeros(len(training_sample))
		target = np.zeros((training_sample.shape[0], training_sample.shape[1]-1, 1 ))
		target[:,0] = 1
		return training_sample, target

	def on_epoch_end(self):
		random.shuffle(self.positive_samples)
		# print ("end of epoch: shuffling")

# import threading

# class threadsafe_iter:
#     """Takes an iterator/generator and makes it thread-safe by
#     serializing call to the `next` method of given iterator/generator.
#     """
#     def __init__(self, it):
#         self.it = it
#         self.lock = threading.Lock()

#     def __iter__(self):
#         return self

#     def next(self):
#         with self.lock:
#             return self.it.next()


# def threadsafe_generator(f):
#     """A decorator that takes a generator function and makes it thread-safe.
#     """
#     def g(*a, **kw):
#         return threadsafe_iter(f(*a, **kw))
# 	return g

# @threadsafe_generator
# def training_generator(positive_samples, negative_samples, alias_dict,
# 	args,):
	
# 	random.seed(args.seed)

# 	batch_size = args.batch_size
# 	num_negative_samples = args.num_negative_samples
# 	n = len(negative_samples)
# 	num_steps = int((len(positive_samples) + batch_size - 1 )/ batch_size)
# 	# I = sp.sparse.csr_matrix(sp.sparse.identity(n))
# 	# I = np.identity(n)

# 	while True:

# 		random.shuffle(positive_samples)

# 		for step in range(num_steps):

# 			batch_positive_samples = np.array(
# 				positive_samples[step * batch_size : (step + 1) * batch_size]).astype(np.int64)
# 			training_sample = get_training_sample(batch_positive_samples, 
# 												  negative_samples, num_negative_samples, alias_dict)
# 			# training_sample = I[training_sample.flatten()].reshape(list(training_sample.shape) + [-1])
# 			yield training_sample, np.zeros(list(training_sample.shape)+[1], dtype=np.int64)