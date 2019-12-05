"""
preprocess() Takes array of variable sized images, process them to be the same size

Input:
	X: n dim array that points to images of variable sizes
Output:
	out: n x r x c array containing zero padded images, where r and c are the max row and max col dimension
		 found among the input images
"""
import numpy as np
# import random
# import matplotlib.pyplot as plt
# X_easy = np.load('TrainImages.npy')
# Y_easy = np.load('TrainY.npy')
# X_hard = np.load('ClassData.npy')
# Y_hard = np.load('ClassLabels.npy')

# No more need because of hardcoding
def get_max_dim(X):
	max_r = max_c = 0
	for i in range(X.shape[0]):
		r, c = X[i].shape
		if r > max_r:
			max_r = r
		if c > max_c:
			max_c = c
	return max_r, max_c

def zero_pad(X):
	#max_r, max_c = get_max_dim(X)
	max_r, max_c = (66, 143) # this is hardcoded from one of the datasets
	#print(max_r, max_c)
	num_img = X.shape[0]
	out = np.zeros((num_img, max_r, max_c))

	for i in range(num_img):
		n_row, n_col = X[i].shape
		if (max_r - n_row) % 2 == 0:
			up_pad = down_pad = int((max_r - n_row) / 2)
		else:
			up_pad = int((max_r - n_row) / 2)
			down_pad = int((max_r - n_row) / 2) + 1

		if (max_c - n_col) % 2 == 0:
			left_pad = right_pad = int((max_c - n_col) / 2)
		else:
			left_pad = int((max_c - n_col) / 2)
			right_pad = int((max_c - n_col) / 2) + 1

		out[i,:,:] = np.pad(X[i], ((up_pad, down_pad), (left_pad, right_pad)), 'constant', constant_values = 0)
	return out
