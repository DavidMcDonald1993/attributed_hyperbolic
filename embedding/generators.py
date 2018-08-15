import random
import numpy as np
import scipy as sp

from utils import get_training_sample


def training_generator(positive_samples, negative_samples, probs,
	num_negative_samples, batch_size=10):
	
	n = len(negative_samples)
	num_steps = int((len(positive_samples) + batch_size - 1 )/ batch_size)
	# I = sp.sparse.csr_matrix(sp.sparse.identity(n))
	# I = np.identity(n)

	while True:

		random.shuffle(positive_samples)

		for step in range(num_steps):

			batch_positive_samples = np.array(
				positive_samples[step * batch_size : (step + 1) * batch_size]).astype(np.int64)
			training_sample = get_training_sample(batch_positive_samples, 
												  negative_samples, num_negative_samples, probs)
			# training_sample = I[training_sample.flatten()].reshape(list(training_sample.shape) + [-1])
			yield training_sample, np.zeros(list(training_sample.shape)+[1], dtype=np.int64)