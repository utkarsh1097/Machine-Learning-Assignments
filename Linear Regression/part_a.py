import numpy as np
import pandas as pd
import sys

#Train and test data path
train_path = sys.argv[1]
test_path = sys.argv[2]

#Output file path
output_path = sys.argv[3]

#Get training data
train_data = pd.read_csv(train_path, header = None)

#Get number of given features and number of training examples
N_train = len(train_data)
M_train = len(train_data.columns) - 1

#Get X_train and Y_train. 
X_train = train_data.iloc[:, 1:].values
Y_train = train_data.iloc[:, 0].values

#Add column of ones to X_train, for the bias unit
X_train = np.append(np.ones((N_train, 1)), X_train, axis = 1)

#Get test data
test_data = pd.read_csv(test_path, header = None)

#Get number of given features and number of test examples
N_test = len(test_data)
M_test = len(test_data.columns) - 1

#Get X_test and Y_test. 
X_test = test_data.iloc[:, 1:].values
Y_test = test_data.iloc[:, 0].values

#Add column of ones to X_test, for the bias unit
X_test = np.append(np.ones((N_test, 1)), X_test, axis = 1)

#Get weights using Analytical Solution
W = np.matmul(np.matmul(np.linalg.pinv(np.matmul(X_train.T, X_train)), X_train.T), Y_train) 

#Get predictions
Y_pred = np.dot(X_test, W)

#Calculate minimum value to be used in Normalized Mean Squared Error
Y_min = np.amin(Y_test)

#Compute the Normalized Mean Squared Error
error = np.dot((Y_pred-Y_test).T, (Y_pred-Y_test))/np.dot((Y_test-Y_min).T, (Y_test-Y_min))

print(error)

#Write the predictions to the output file specified
file = open(output_path, 'w')
for i in range(N_test):
	file.writelines(str(Y_pred[i])+'\n')
file.close()