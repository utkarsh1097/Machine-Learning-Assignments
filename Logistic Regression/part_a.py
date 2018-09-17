import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, vstack
import sys
import math
import re

def sigmoid(Z):	#Sigmoid function
	return np.exp(Z)/(1+np.exp(Z))

def predictions(weights, X):	#Given W and X, return predictions
	return sigmoid(csr_matrix.dot(X, weights))	

def calc_gradient(X, e):
	return csr_matrix.dot(X.T, e)

def gradient_descent(weights, X_train, Y_train, M_data, alpha, lambd):	#One step of gradient descent, so that it can be used across all the three methods
	N_train = Y_train.shape[0]	#Number of training examples

	pred_error = predictions(weights, X_train) - Y_train	#(n*(2m-1)+2n)+n
	gradient = calc_gradient(X_train, pred_error)	#m*(2n-1)

	weights = weights*(1-alpha*lambd) - (alpha*gradient)	#m+(2)+m+m

	return weights

def constant_learning_rate(lambd_list, X_data, Y_data, M_data, N_data, alpha, num_iters = 500):
	best_lambd = -1
	min_error = 100000000
	
	#We will do a 10-fold cross validation for all values of lambda, and then choose the best one
	for lambd in lambd_list:
		
		mean_error = 0	#Fraction of incorrect predictions
		fold_size = N_data//10

		for i in range(10):
			#Train and CV split
			X_train = vstack((X_data[0:i*fold_size], X_data[(i+1)*fold_size:N_data]), format = 'csr')
			X_cv = X_data[i*fold_size:(i+1)*fold_size]

			Y_train = np.append(Y_data[0:i*fold_size], Y_data[(i+1)*fold_size:N_data], axis = 0)
			Y_cv = Y_data[i*fold_size:(i+1)*fold_size]

			N_cv = Y_cv.shape[0]	#Will be useful for finding error			
			weights = np.zeros((M_data, 1))	#Initialize
			
			#Now train for num_iter iterations on the train data 
			for i in range(num_iters):
				weights = gradient_descent(weights, X_train, Y_train, M_data, alpha, lambd)	#After 500 steps of iterations

			Y_pred = predictions(weights, X_cv) > 0.5
			Y_pred = Y_pred.astype(int)	#Y_pred was originally boolean

			error_perc = 1-((np.sum(Y_pred == Y_cv))/N_cv)
			# print(error_perc, (np.sum(Y_pred == Y_cv)), N_cv, Y_pred.shape, Y_cv.shape)
			
			mean_error += error_perc

		mean_error/=10
		# print("Mean error = ", mean_error)
		if(mean_error < min_error):
			min_error = mean_error
			best_lambd = lambd

	return best_lambd


def adaptive_learning_rate(lambd_list, X_data, Y_data, M_data, N_data, alpha, num_iters = 500):
	best_lambd = -1
	min_error = 100000000
	
	#We will do a 10-fold cross validation for all values of lambda, and then choose the best one
	for lambd in lambd_list:
		
		mean_error = 0	#Fraction of incorrect predictions
		fold_size = N_data//10

		for i in range(10):
			#Train and CV split
			X_train = vstack((X_data[0:i*fold_size], X_data[(i+1)*fold_size:N_data]), format = 'csr')
			X_cv = X_data[i*fold_size:(i+1)*fold_size]

			Y_train = np.append(Y_data[0:i*fold_size], Y_data[(i+1)*fold_size:N_data], axis = 0)
			Y_cv = Y_data[i*fold_size:(i+1)*fold_size]

			N_cv = Y_cv.shape[0]	#Will be useful for finding error	
			weights = np.zeros((M_data, 1))	#Initialize
			
			#Now train for num_iter iterations on the train data 
			for j in range(num_iters):
				weights = gradient_descent(weights, X_train, Y_train, M_data, (alpha)/math.sqrt(j+1), lambd)	#After 500 steps of iterations

			Y_pred = predictions(weights, X_cv) > 0.5
			Y_pred = Y_pred.astype(int)	#Y_pred was originally boolean

			error_perc = 1-((np.sum(Y_pred == Y_cv))/N_cv)
			# print(error_perc, (np.sum(Y_pred == Y_cv)), N_cv, Y_pred.shape, Y_cv.shape)
			
			mean_error += error_perc

		mean_error/=10
		# print("Mean error = ", mean_error)
		if(mean_error < min_error):
			min_error = mean_error
			best_lambd = lambd

	return best_lambd

def derivative_f(alpha, d, X, e):
	Xd = csr_matrix.dot(X, d)	#n*(2m-1)
	derivative = 2*alpha*np.dot(Xd.T, Xd) - 2*np.dot(Xd.T, e)	#(1+1+1*(2m-1)) + (1+1*(2m-1)) 
	return derivative

def exact_line_search(lambd_list, X_data, Y_data, M_data, N_data, alpha, num_iters = 100):
	best_lambd = -1
	min_error = 100000000
	
	#We will do a 10-fold cross validation for all values of lambda, and then choose the best one
	for lambd in lambd_list:
		
		mean_error = 0	#Fraction of incorrect predictions
		fold_size = N_data//10

		for i in range(10):
			#Train and CV split
			X_train = vstack((X_data[0:i*fold_size], X_data[(i+1)*fold_size:N_data]), format = 'csr')
			X_cv = X_data[i*fold_size:(i+1)*fold_size]

			Y_train = np.append(Y_data[0:i*fold_size], Y_data[(i+1)*fold_size:N_data], axis = 0)
			Y_cv = Y_data[i*fold_size:(i+1)*fold_size]

			N_cv = Y_cv.shape[0]	#Will be useful for finding error
			weights = np.zeros((M_data, 1))	#Initialize
			
			#Now train for num_iter iterations on the train data 
			for i in range(num_iters):
				#This time, we also binary-search for the optimum rate
				pred_error = predictions(weights, X_train) - Y_train
				d = calc_gradient(X_train, pred_error) #vector along optimum

				low_alpha = 0	#Lower bound of alpha
				high_alpha = alpha 	#Upperbound of alpha
				while(derivative_f(high_alpha, d, X_train, pred_error) < 0):
					high_alpha = 2*high_alpha

				#Now both bounds have been found, so start the binary search
				use_alpha = -1
				
				while(True):
					mid_alpha = (low_alpha+high_alpha)/2
					cur_derivative = derivative_f(mid_alpha, d, X_train, pred_error)
					if(abs(cur_derivative) < 0.0001):	#tolerance
						use_alpha = mid_alpha
						break
					else:
						if(cur_derivative > 0):
							high_alpha = mid_alpha
						else:
							low_alpha = mid_alpha

				weights = gradient_descent(weights, X_train, Y_train, M_data, use_alpha, lambd)

			Y_pred = predictions(weights, X_cv) > 0.5
			Y_pred = Y_pred.astype(int)	#Y_pred was originally boolean

			error_perc = 1-((np.sum(Y_pred == Y_cv))/N_cv)
			# print(error_perc, (np.sum(Y_pred == Y_cv)), N_cv, Y_pred.shape, Y_cv.shape)
			
			mean_error += error_perc

		mean_error/=10
		# print("Mean error = ", mean_error)
		if(mean_error < min_error):
			min_error = mean_error
			best_lambd = lambd

	return best_lambd

def likelihood_function(weights, X_data, Y_data, lambd):
	temp1 = np.dot(Y_data.T, np.log(predictions(weights, X_data)))
	temp2 = np.dot((1.0-Y_data).T, np.log(1.0-predictions(weights, X_data)))
	cost = (temp1 + temp2) + (lambd*np.dot(weights.T, weights))/2
	print(cost[0, 0])
	return cost[0, 0]

def pred_constant_learning_rate(best_lambd, X_data, Y_data, X_test, M_data, alpha, num_iters = 500):
	weights = np.zeros((M_data, 1))	#Initialize

	for i in range(num_iters):
		weights = gradient_descent(weights, X_data, Y_data, M_data, alpha, best_lambd)
		
	Y_pred = predictions(weights, X_test) > 0.5
	Y_pred = Y_pred.astype(int)	#Y_pred was originally boolean

	return Y_pred


def pred_adaptive_learning_rate(best_lambd, X_data, Y_data, X_test, M_data, alpha, num_iters = 500):
	weights = np.zeros((M_data, 1))	#Initialize

	for i in range(num_iters):
		weights = gradient_descent(weights, X_data, Y_data, M_data, alpha/np.sqrt(i+1), best_lambd)
		
	Y_pred = predictions(weights, X_test) > 0.5
	Y_pred = Y_pred.astype(int)	#Y_pred was originally boolean

	return Y_pred

def pred_exact_line_search(best_lambd, X_data, Y_data, X_test, M_data, alpha, num_iters = 100):
	weights = np.zeros((M_data, 1))	#Initialize

	N_data = Y_data.shape[0]

	for i in range(num_iters):

		pred_error = predictions(weights, X_data) - Y_data
		d = calc_gradient(X_data, pred_error)

		low_alpha = 0	#Lower bound of alpha
		high_alpha = alpha 	#Upperbound of alpha
				
		while(derivative_f(high_alpha, d, X_data, pred_error) < 0):
			high_alpha = 2*high_alpha

		#Now both bounds have been found, so start the binary search
		use_alpha = -1
		
		while(True):
			mid_alpha = (low_alpha+high_alpha)/2
			
			cur_derivative = derivative_f(mid_alpha, d, X_data, pred_error)
			
			if(abs(cur_derivative) < 0.0001):	#tolerance
				use_alpha = mid_alpha
				break
			else:
				if(cur_derivative > 0):
					high_alpha = mid_alpha
				else:
					low_alpha = mid_alpha

		weights = gradient_descent(weights, X_data, Y_data, M_data, use_alpha, best_lambd)

	Y_pred = predictions(weights, X_test) > 0.5
	Y_pred = Y_pred.astype(int)	#Y_pred was originally boolean

	return Y_pred

def main():

	method = sys.argv[1]
	alpha = float(sys.argv[2])
	num_iters = int(sys.argv[3])
	batch_size = int(sys.argv[4])
	data_path = sys.argv[5]
	vocab_path = sys.argv[6]
	test_path = sys.argv[7]
	out_path = sys.argv[8]

	vocab_file = open(vocab_path)
	vocab = vocab_file.read().splitlines()
	vocab_dict = dict()
	for i in range(len(vocab)):
		vocab_dict[vocab[i].lower()] = i+1	#idx = 0 not being used, reserved for the bias term. convert all vocab terms to lower

	#Train+CV data
	data_df = pd.read_csv(data_path, header = None)
	N_data = len(data_df)
	M_data = len(vocab)+1	#One extra will be for the bias term
	X_data = []
	Y_data = data_df.iloc[:, 0].values
	Y_data = np.asmatrix(Y_data).reshape(N_data, 1)

	#Test data
	test_df = pd.read_csv(test_path, header = None)
	N_test = len(test_df)
	M_test = len(vocab)+1
	X_test = []
	Y_test = test_df.iloc[:, 0].values
	Y_test = np.asmatrix(Y_test).reshape(N_test, 1)

	# Construct X_data
	for i in range(len(data_df)):
		features = np.zeros((1, M_data))
		features[0][0] = 1	#bias term set at 1
		sentence = data_df.iloc[i][1].lower()	#convert sentence to lowercase
		
		#These features aren't accounted for otherwise
		features[0][vocab_dict['?']] = sentence.count('?')
		features[0][vocab_dict['!']] = sentence.count('!')
		features[0][vocab_dict[';)']] = sentence.count(';)')
		features[0][vocab_dict[';-)']] = sentence.count(';-)')
		features[0][vocab_dict[');']] = sentence.count(');')
		features[0][vocab_dict[';d']] = sentence.count(';d')
		features[0][vocab_dict[';o)']] = sentence.count(';o)')
		features[0][vocab_dict[';p']] = sentence.count(';p')
		features[0][vocab_dict[':)']] = sentence.count(':)')
		features[0][vocab_dict[':-)']] = sentence.count(':-)')
		features[0][vocab_dict['=)']] = sentence.count('=)')
		features[0][vocab_dict['8)']] = sentence.count('8)')
		features[0][vocab_dict['):']] = sentence.count('):')
		features[0][vocab_dict[':o)']] = sentence.count(':o)')
		features[0][vocab_dict['=o)']] = sentence.count('=o)')
		features[0][vocab_dict[':(']] = sentence.count(':(')
		features[0][vocab_dict['(8']] = sentence.count('(8')
		features[0][vocab_dict[':-(']] = sentence.count(':-(')
		features[0][vocab_dict['(=']] = sentence.count('(=')
		features[0][vocab_dict['8(']] = sentence.count('8(')
		features[0][vocab_dict['=(']] = sentence.count('=(')
		features[0][vocab_dict['=d']] = sentence.count('=d')
		features[0][vocab_dict['=]']] = sentence.count('=]')
		features[0][vocab_dict['=p']] = sentence.count('=p')
		features[0][vocab_dict['[=']] = sentence.count('[=')
		features[0][vocab_dict[':d']] = sentence.count(':d')
		features[0][vocab_dict[':p']] = sentence.count(':p')
		features[0][vocab_dict['d:']] = sentence.count('d:')
		features[0][vocab_dict[':-d']] = sentence.count(':-d')
		features[0][vocab_dict['8:']] = sentence.count('8:')
		features[0][vocab_dict[':-p']] = sentence.count(':-p')
		features[0][vocab_dict[':}']] = sentence.count(':}')

		sentence = re.sub("[^0-9a-zA-Z '-]", '', sentence)
		sentence = sentence.split(' ')
		for j in range(len(sentence)):
			if(sentence[j] in vocab_dict.keys()):
				features[0][vocab_dict[sentence[j]]] += 1
		X_data.append(csr_matrix(features))

	X_data = vstack(X_data, format = 'csr')

	# print("Train+CV data prepared")

	#Construct X_test
	for i in range(len(test_df)):
		features = np.zeros((1, M_test))
		features[0][0] = 1	#bias term set at 1
		sentence = test_df.iloc[i][1].lower()	#convert sentence to lowercase

		#These features aren't accounted for otherwise
		features[0][vocab_dict['?']] = sentence.count('?')
		features[0][vocab_dict['!']] = sentence.count('!')
		features[0][vocab_dict[';)']] = sentence.count(';)')
		features[0][vocab_dict[';-)']] = sentence.count(';-)')
		features[0][vocab_dict[');']] = sentence.count(');')
		features[0][vocab_dict[';d']] = sentence.count(';d')
		features[0][vocab_dict[';o)']] = sentence.count(';o)')
		features[0][vocab_dict[';p']] = sentence.count(';p')
		features[0][vocab_dict[':)']] = sentence.count(':)')
		features[0][vocab_dict[':-)']] = sentence.count(':-)')
		features[0][vocab_dict['=)']] = sentence.count('=)')
		features[0][vocab_dict['8)']] = sentence.count('8)')
		features[0][vocab_dict['):']] = sentence.count('):')
		features[0][vocab_dict[':o)']] = sentence.count(':o)')
		features[0][vocab_dict['=o)']] = sentence.count('=o)')
		features[0][vocab_dict[':(']] = sentence.count(':(')
		features[0][vocab_dict['(8']] = sentence.count('(8')
		features[0][vocab_dict[':-(']] = sentence.count(':-(')
		features[0][vocab_dict['(=']] = sentence.count('(=')
		features[0][vocab_dict['8(']] = sentence.count('8(')
		features[0][vocab_dict['=(']] = sentence.count('=(')
		features[0][vocab_dict['=d']] = sentence.count('=d')
		features[0][vocab_dict['=]']] = sentence.count('=]')
		features[0][vocab_dict['=p']] = sentence.count('=p')
		features[0][vocab_dict['[=']] = sentence.count('[=')
		features[0][vocab_dict[':d']] = sentence.count(':d')
		features[0][vocab_dict[':p']] = sentence.count(':p')
		features[0][vocab_dict['d:']] = sentence.count('d:')
		features[0][vocab_dict[':-d']] = sentence.count(':-d')
		features[0][vocab_dict['8:']] = sentence.count('8:')
		features[0][vocab_dict[':-p']] = sentence.count(':-p')
		features[0][vocab_dict[':}']] = sentence.count(':}')

		sentence = re.sub("[^0-9a-zA-Z '-]", '', sentence)
		sentence = sentence.split(' ')
		for j in range(len(sentence)):
			if(sentence[j] in vocab_dict.keys()):
				features[0][vocab_dict[sentence[j]]] += 1
		X_test.append(csr_matrix(features))

	X_test = vstack(X_test, format = 'csr')

	# print("Test data prepared")

	#Now, feature matrix has been created. Let's start gradient descent for now
	lambd_list = [0, 0.05, 0.1, 0.2, 0.5, 0.7, 1.0]

	if(method == '1'):	#Constant learning rate
		best_lambd = constant_learning_rate(lambd_list, X_data, Y_data, M_data, N_data, alpha, num_iters)
		Y_pred = pred_constant_learning_rate(best_lambd, X_data, Y_data, X_test, M_data, alpha, num_iters)

		out_file = open(out_path, 'w')
		for i in range(len(Y_pred)):
			out_file.write(str(Y_pred[i, 0])+'\n')

	elif(method == '2'):
		best_lambd = adaptive_learning_rate(lambd_list, X_data, Y_data, M_data, N_data, alpha, num_iters)
		Y_pred = pred_adaptive_learning_rate(best_lambd, X_data, Y_data, X_test, M_data, alpha, num_iters)

		out_file = open(out_path, 'w')
		for i in range(len(Y_pred)):
			out_file.write(str(Y_pred[i, 0])+'\n')
			
	else:
		best_lambd = exact_line_search(lambd_list, X_data, Y_data, M_data, N_data, alpha, num_iters)
		Y_pred = pred_exact_line_search(best_lambd, X_data, Y_data, X_test, M_data, alpha, num_iters)

		out_file = open(out_path, 'w')
		for i in range(len(Y_pred)):
			out_file.write(str(Y_pred[i, 0])+'\n')

if __name__ == '__main__':
	main()