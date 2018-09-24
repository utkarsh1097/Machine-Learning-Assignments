import numpy as np
import pandas as pd
import sys

np.random.seed(0)

#Obtain the OHE of input vector
def one_hot_encoder(Y):
	new_Y = np.zeros((Y.shape[0], np.max(Y)+1))
	new_Y[np.arange(Y.shape[0]), Y] = 1
	return new_Y

#Softmax function
def softmax(Z):
	N, M = Z.shape
	val = np.exp(Z - (np.max(Z, axis = 1).reshape(N, 1)))
	return val/(np.sum(val, axis = 1)).reshape(N, 1)

#Sigmoid function
def sigmoid(Z):
	return 1/(1+np.exp(-Z))
	
# Derivative of softmax already handled in backwardpropagation()
def derivative_sigmoid(A):
	return (A*(1-A))

#Initialize weights and bias for the network
def init_network(M_data, num_labels, num_hidden_layers, hidden_layer_sizes):
	W = dict()	#W[l] means the weights between layer l and layer l-1. Dimension = (#units in layer l-1 x #units in layer l)
	b = dict() #b[l] means the bias value added for each unit of layer l
	
	#Will use 1-based indexing for weights, since hidden-layer are also 1-based indexed	
	L = num_hidden_layers

	#Will use Xavier initialization of weights
	W[1] = np.random.randn(M_data, hidden_layer_sizes[0])*np.sqrt(1.0/M_data)	
	b[1] = np.zeros((1, hidden_layer_sizes[0]))
	
	for i in range(1, L):
		W[i+1] = np.random.randn(hidden_layer_sizes[i-1], hidden_layer_sizes[i])*np.sqrt(1.0/hidden_layer_sizes[i-1])
		b[i+1] = np.zeros((1, hidden_layer_sizes[i]))
			
	W[L+1] = np.random.randn(hidden_layer_sizes[L-1], num_labels)*np.sqrt(1.0/hidden_layer_sizes[L-1])	#since output layer is necessarily softmax
	b[L+1] = np.zeros((1, num_labels))

	return W, b

# Forward propagation function
def forwardpropagation(W, b, X_data, num_hidden_layers):
	A = dict()	#Output of all layers. Output of input layer = a[0]

	L = num_hidden_layers
	A[0] = X_data
	
	for i in range(L):
		Z = np.dot(A[i], W[i+1]) + b[i+1]
		A[i+1] = sigmoid(Z)
		
	Z = np.dot(A[L], W[L+1]) + b[L+1]
	A[L+1] = softmax(Z)

	return A

#Backward propagation function
def backwardpropagation(A, W, num_hidden_layers, num_examples, Y_data):
	dW = dict()	#dW[l] means dL/dW[l] corresponding to weights for that layer
	db = dict()	#db[l] means dL/db[l] corresponding to biases for that layer

	L = num_hidden_layers
	
	dZ = (A[L+1] - Y_data)/(num_examples)	#Handled softmax's derivative
	dW[L+1] = np.dot(A[L].T, dZ)
	db[L+1] = np.sum(dZ, axis = 0, keepdims = True)
	
	for i in range(L, 0, -1):
		dZ = np.dot(dZ, W[i+1].T)*derivative_sigmoid(A[i])
		dW[i] = np.dot(A[i-1].T, dZ)
		db[i] = np.sum(dZ, axis = 0, keepdims = True)

	return dW, db

#Updating the weights using the gradients
def gradientdescent(W, b, dW, db, learning_rate, num_hidden_layers):
	for i in range(num_hidden_layers+1):
		W[i+1] = W[i+1] - learning_rate*dW[i+1]
		b[i+1] = b[i+1] - learning_rate*db[i+1]

	return W, b

#Loss function used: cross entropy
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
	initial_lr = 1.4	#Final initial learning rate
	hidden_layer_sizes = [500, 300]	#Final hidden layers
	num_hidden_layers = len(hidden_layer_sizes)
	
	#Get train data
	train_df = pd.read_csv(train_path, header = None)

	X_train = train_df.iloc[:, 1:].values
	Y_train = train_df.iloc[:, 0].values 
	Y_train = one_hot_encoder(Y_train)
	num_labels = Y_train.shape[1]
	N_train, M_train = X_train.shape

	#Normalize train data
	X_train = (X_train)/np.max(X_train)	#Divide by max pixel value

	#Get test data
	test_df = pd.read_csv(test_path, header = None)
	N_test = len(test_df)
	X_test = test_df.iloc[:, 1:].values
	M_test = X_test.shape[1]

	#Normalize test data
	X_test = (X_test)/np.max(X_train)	#Divide by max pixel value

	#Initialize weights for the neural network. Note: M_train == M_test == M_data
	W, b = init_network(M_train, num_labels, num_hidden_layers, hidden_layer_sizes)

	num_epochs = 100	#Training for 100 epochs
	
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
			A = forwardpropagation(W, b, X_train_batch[j], num_hidden_layers)
			dW, db = backwardpropagation(A, W, num_hidden_layers, X_train_batch[j].shape[0], Y_train_batch[j])
			
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
	A = forwardpropagation(W, b, X_test, num_hidden_layers)

	Y_pred = np.argmax(A[num_hidden_layers+1], axis = 1)

	out_file = open(out_path, 'w')
	for i in range(len(Y_pred)):
		out_file.write(str(Y_pred[i])+"\n")
	out_file.close()

if __name__ == '__main__':
	main()