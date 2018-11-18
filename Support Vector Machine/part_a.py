import numpy as np
import pandas as pd
from cvxopt import matrix, solvers
from sklearn.model_selection import train_test_split
import sys

np.random.seed(0)

def load_data(path):
	train_df = pd.read_csv(path, header = None)
	X_train = train_df.iloc[:, 1:].values
	X_train = X_train/255
	Y_train = train_df.iloc[:, 0].values
	Y_train[Y_train == 0] = -1	#Since SVMs deal in y = {-1, 1}
	Y_train = Y_train.reshape(len(Y_train), 1)	#Nx1 shape will be useful in this assignment

	return X_train, Y_train

def get_train_val_data(X_data, Y_data, ratio):
	X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size = ratio)
	return X_train, X_val, Y_train, Y_val

def solve_dual(X_train, Y_train, N, C):
	print(np.dot(Y_train, Y_train.T)*np.dot(X_train, X_train.T))
	Q = matrix(np.dot(Y_train, Y_train.T)*np.dot(X_train, X_train.T))
	p = matrix(-np.ones(N))
	G = matrix(np.vstack((-np.identity(N), np.identity(N))))
	h = matrix(np.vstack((np.zeros((N, 1)), np.ones((N, 1))*C)))
	A = matrix(Y_train.reshape(1, N)*1.0)
	B = matrix(np.array([0.0]))

	opt = solvers.qp(Q, p, G, h, A, B)
	alpha = np.array(opt['x'])

	return alpha

def get_weights(alpha, X_train, Y_train):	
	W = np.dot(X_train.T, Y_train*alpha)
	b = np.mean(Y_train - np.dot(X_train, W))

	return W, b

def get_predictions(X_test, W, b):
	Y_pred = np.dot(X_test, W)+b
	Y_pred[Y_pred >= 0] = 1
	Y_pred[Y_pred < 0] = 0

	return Y_pred

def main():
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]
	C = float(sys.argv[4])

	X_train, Y_train = load_data(train_path)
	N = len(X_train)

	alpha = solve_dual(X_train, Y_train, N, C)
	W, b = get_weights(alpha, X_train, Y_train)

	X_test, Y_test = load_data(test_path)
	
	Y_pred = get_predictions(X_test, W, b)

	out_file = open(out_path, 'w')
	for i in range(len(Y_pred)):
		out_file.write(str(int(Y_pred[i][0]))+'\n')
	out_file.close()

if __name__ == '__main__':
	main()


