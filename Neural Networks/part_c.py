import numpy as np
import pandas as pd
import sys
from skimage.filters import gabor

np.random.seed(0)

def one_hot_encoder(Y):
	new_Y = np.zeros((Y.shape[0], np.max(Y)+1))
	new_Y[np.arange(Y.shape[0]), Y] = 1
	return new_Y

#Sigmoid function
def sigmoid(Z):
	return 1/(1+np.exp(-Z))

#Softmax function
def softmax(Z):
	N, M = Z.shape
	val = np.exp(Z - (np.max(Z, axis = 1).reshape(N, 1)))
	return val/(np.sum(val, axis = 1)).reshape(N, 1)

#Tanh function, can be written in terms of sigmoid
def tanh(Z):
	return np.tanh(Z)

#Relu function
def relu(Z):
	return Z*(Z > 0)

def activate(Z, activation):
	if(activation == 'sigmoid'):
		return sigmoid(Z)
	elif(activation == 'relu'):
		return relu(Z)
	elif(activation == 'softmax'):
		return softmax(Z)
	else:
		return tanh(Z)

def derivative_activate(A, activation):
	# Derivative of softmax already handled in backwardpropagation()
	if(activation == 'sigmoid'):
		return (A*(1-A))
	elif(activation == 'relu'):
		return ((A>0)).astype(int)
	else:	
		return (1-A*A)


def init_network(M_data, num_labels, num_hidden_layers, hidden_layer_sizes, activation):
	W = dict()	#W[l] means the weights between layer l and layer l-1. Dimension = (#units in layer l-1 x #units in layer l)
	b = dict() #b[l] means the bias value added for each unit of layer l
	#Will use 1-based indexing for weights, since hidden-layer are also 1-based indexed	
	L = num_hidden_layers

	#Will use Xavier initialization of weights
	if(activation == 'relu'):
		W[1] = np.random.randn(M_data, hidden_layer_sizes[0])*np.sqrt(2.0/M_data)	#Factor of 2 helps in case of relu activation function
	else:
		W[1] = np.random.randn(M_data, hidden_layer_sizes[0])*np.sqrt(1.0/M_data)	
	b[1] = np.zeros((1, hidden_layer_sizes[0]))
	for i in range(1, L):
		if(activation == 'relu'):
			W[i+1] = np.random.randn(hidden_layer_sizes[i-1], hidden_layer_sizes[i])*np.sqrt(2.0/hidden_layer_sizes[i-1])
		else:
			W[i+1] = np.random.randn(hidden_layer_sizes[i-1], hidden_layer_sizes[i])*np.sqrt(1.0/hidden_layer_sizes[i-1])
		b[i+1] = np.zeros((1, hidden_layer_sizes[i]))
			
	W[L+1] = np.random.randn(hidden_layer_sizes[L-1], num_labels)*np.sqrt(1.0/hidden_layer_sizes[L-1])	#since output layer is necessarily softmax
	b[L+1] = np.zeros((1, num_labels))

	return W, b

# Forward propagation function
def forwardpropagation(W, b, X_data, num_hidden_layers, activation):
	A = dict()	#Output of all layers. Output of input layer = a[0]

	L = num_hidden_layers
	A[0] = X_data
	
	for i in range(1, L+1):
		Z = np.dot(A[i-1], W[i]) + b[i]
		A[i] = activate(Z, activation)
		
	Z = np.dot(A[L], W[L+1]) + b[L+1]
	A[L+1] = activate(Z, 'softmax')

	return A

def backwardpropagation(A, W, num_hidden_layers, num_examples, Y_data, activation):
	dW = dict()
	db = dict()

	L = num_hidden_layers
	
	dZ = (A[L+1] - Y_data)/(num_examples)
	dW[L+1] = np.dot(A[L].T, dZ)
	db[L+1] = np.sum(dZ, axis = 0, keepdims = True)
	
	for i in range(L, 0, -1):
		dZ = np.dot(dZ, W[i+1].T)*derivative_activate(A[i], activation)
		dW[i] = np.dot(A[i-1].T, dZ)
		db[i] = np.sum(dZ, axis = 0, keepdims = True)

	return dW, db

def gradientdescent(W, b, dW, db, learning_rate, num_hidden_layers):
	for i in range(num_hidden_layers+1):
		W[i+1] = W[i+1] - learning_rate*dW[i+1]
		b[i+1] = b[i+1] - learning_rate*db[i+1]

	return W, b

def loss_function(Y_pred, Y, num_examples):
	logY_pred = np.log(Y_pred)
	loss = -(np.sum(Y*logY_pred))/num_examples
	return loss

def main():
	#Command line arguments
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]
	batch_size = 100 #Final batch size
	initial_lr = 0.2	#Final initial learning rate
	hidden_layer_sizes = [500, 300]	#Final hidden layers
	num_hidden_layers = len(hidden_layer_sizes)
	activation = 'relu'
	
	#Get train data
	train_df = pd.read_csv(train_path, header = None)

	X_train = train_df.iloc[:, 1:].values
	Y_train = train_df.iloc[:, 0].values 
	Y_train_orig = Y_train
	Y_train = one_hot_encoder(Y_train)
	num_labels = Y_train.shape[1]
	N_train, M_train = X_train.shape

	
	for i in range(N_train):
		filt_real1, _ = gabor(X_train[i].reshape((32, 32)), frequency=0.25, theta=np.pi/4, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
		filt_real2, _ = gabor(X_train[i].reshape((32, 32)), frequency=0.25, theta=np.pi/2, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
		filt_real3, _ = gabor(X_train[i].reshape((32, 32)), frequency=0.25, theta=3*np.pi/4, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
		filt_real4, _ = gabor(X_train[i].reshape((32, 32)), frequency=0.25, theta=np.pi, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
	
		X_train[i] = np.maximum(filt_real1.reshape(1024), np.maximum(filt_real2.reshape(1024), np.maximum(filt_real3.reshape(1024), filt_real4.reshape(1024))))
		
	#Normalize train data
	X_train = (X_train)/np.max(X_train)	#Divide by max pixel value

	#Get test data
	test_df = pd.read_csv(test_path, header = None)
	N_test = len(test_df)
	X_test = test_df.iloc[:, 1:].values
	M_test = X_test.shape[1]

	for i in range(N_test):
		filt_real1, _ = gabor(X_test[i].reshape((32, 32)), frequency=0.25, theta=np.pi/4, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
		filt_real2, _ = gabor(X_test[i].reshape((32, 32)), frequency=0.25, theta=np.pi/2, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
		filt_real3, _ = gabor(X_test[i].reshape((32, 32)), frequency=0.25, theta=3*np.pi/4, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
		filt_real4, _ = gabor(X_test[i].reshape((32, 32)), frequency=0.25, theta=np.pi, sigma_x=1, sigma_y=1, offset=np.pi/5, n_stds=3)
	
		X_test[i] = np.maximum(filt_real1.reshape(1024), np.maximum(filt_real2.reshape(1024), np.maximum(filt_real3.reshape(1024), filt_real4.reshape(1024))))

	#Normalize test data
	X_test = (X_test)/np.max(X_train)	#Divide by max pixel value

	#Initialize weights for the neural network. Note: M_train == M_test == M_data
	W, b = init_network(M_train, num_labels, num_hidden_layers, hidden_layer_sizes, activation)

	num_epochs = 120	#This is being set beforehand
	
	batch_idx = 1	#will help us track the current batch
	num_batches = 0	#number of minibatches 

	#Now, construct our minibatches
	X_train_batch = dict()
	Y_train_batch = dict()

	while batch_idx != N_train:
		
		if(batch_idx+batch_size < N_train):
			X_train_batch[num_batches] = X_train[batch_idx:batch_idx+batch_size]
			Y_train_batch[num_batches] = Y_train[batch_idx:batch_idx+batch_size]
			batch_idx = batch_idx+batch_size
		else:
			X_train_batch[num_batches] = X_train[batch_idx:N_train]
			Y_train_batch[num_batches] = Y_train[batch_idx:N_train]
			batch_idx = N_train

		num_batches += 1
			

	#Below values are meant for adaptive learning rate
	cur_avg_loss = 0
	prev_avg_loss = 1000000
	adaptive_factor = 1

	cur_lr = initial_lr

	for i in range(num_epochs):	#For now, number of iterations = 1000
		for j in range(num_batches):
			A = forwardpropagation(W, b, X_train_batch[j], num_hidden_layers, activation)
			dW, db = backwardpropagation(A, W, num_hidden_layers, X_train_batch[j].shape[0], Y_train_batch[j], activation)
			
			W, b = gradientdescent(W, b, dW, db, cur_lr, num_hidden_layers)
			cur_loss = loss_function(A[num_hidden_layers+1], Y_train_batch[j], X_train_batch[j].shape[0])
		
			cur_avg_loss += cur_loss
		
		cur_avg_loss = cur_avg_loss/num_batches
		if(cur_avg_loss > prev_avg_loss):
			adaptive_factor += 1
			cur_lr = initial_lr/np.sqrt(adaptive_factor)
		print("Epoch " + str(i+1) + " done! Loss = "+str(cur_avg_loss))
		prev_avg_loss = cur_avg_loss
		cur_avg_loss = 0

	#Time to make predictions
	A = forwardpropagation(W, b, X_test, num_hidden_layers, activation)

	Y_pred = np.argmax(A[num_hidden_layers+1], axis = 1)

	out_file = open(out_path, 'w')
	for i in range(len(Y_pred)):
		out_file.write(str(Y_pred[i])+"\n")
	out_file.close()

if __name__ == '__main__':
	main()