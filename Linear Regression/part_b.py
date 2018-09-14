import pandas as pd
import numpy as np
import sys

#(Train+CV) data path and test data path
data_path = sys.argv[1]
test_path = sys.argv[2]

#Output file path
output_path = sys.argv[3]

#Get (Train+CV) data
data = pd.read_csv(data_path, header = None)

#Get number of given features and number of training examples
N_data = len(data)
M_data = len(data.columns) - 1

#Get X_data and Y_data. 
X_data = data.iloc[:, 1:].values
Y_data = data.iloc[:, 0].values

#Add column of ones to X_data, for the bias unit
X_data = np.append(np.ones((N_data, 1)), X_data, axis = 1)

#Get test data
test_data = pd.read_csv(test_path, header = None)

#Get number of given features and number of test examples
N_test = len(test_data)
M_test = len(test_data.columns) - 1

#Get X_test and Y_test. Reshape them accordingly
X_test = test_data.iloc[:, 1:].values
Y_test = test_data.iloc[:, 0].values

#Add column of ones to X_test, for the bias unit
X_test = np.append(np.ones((N_test, 1)), X_test, axis = 1)


#We will store the best value of lambda, which is our hyperparameter
best_lambd = -1
min_error = 1000000000000

for i in range(0, 1001, 50):
	#Initialize lambda, fold size and mean error
	lambd = i/1000
	fold_size = N_data//10
	mean_error = 0

	#Do a 10-fold cross validation
	for i in range(0, 10):
		#Train and CV input split
		X_train = np.append(X_data[0:i*fold_size], X_data[(i+1)*fold_size:N_data], axis = 0)

		X_cv = X_data[i*fold_size:(i+1)*fold_size]

		#Train and CV output split
		Y_train = np.append(Y_data[0:i*fold_size], Y_data[(i+1)*fold_size:N_data], axis = 0)
		Y_cv = Y_data[i*fold_size:(i+1)*fold_size]

		#Get weights using Analytical Solution
		W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_train.T, X_train) + lambd*np.identity(M_data+1)), X_train.T), Y_train) 
		print("LOL", )
		#Get predictions
		Y_pred = np.dot(X_cv, W)

		#Calculate minimum value to be used in Normalized Mean Squared Error
		Y_min = np.amin(Y_cv)

		#Compute the Normalized Mean Squared Error
		error = np.dot((Y_pred-Y_cv).T, (Y_pred-Y_cv))/np.dot((Y_cv-Y_min).T, (Y_cv-Y_min))

		#Update mean_error
		mean_error += error

	mean_error/=10

	#If mean error is less than min error, choose current lambda as the best lambda
	if(mean_error < min_error):
		min_error = mean_error
		best_lambd = lambd


#Now using the best value of lambda let's get the model
final_W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_data.T, X_data) + best_lambd*np.identity(M_data+1)), X_data.T), Y_data)

#Final prediction
final_Y_pred = np.dot(X_test, final_W)

#Calculate minimum value to be used in Normalized Mean Squared Error
Y_min = np.amin(Y_test)

#Compute the Normalized Mean Squared Error
error = np.dot((final_Y_pred-Y_test).T, (final_Y_pred-Y_test))/np.dot((Y_test-Y_min).T, (Y_test-Y_min))

#Print error for convenience
print(error, best_lambd)

#Write the predictions to the output file specified
file = open(output_path, 'w')
for i in range(N_test):
	file.writelines(str(final_Y_pred[i])+'\n')
file.close()