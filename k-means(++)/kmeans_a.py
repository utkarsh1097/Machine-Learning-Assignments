import numpy as np
import sys
import random
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

#Function to load the data
def get_data(train_path):
	train_data = np.load(train_path)
	X_data = train_data[:, 1:]
	Y_data = train_data[:, 0]
	X_data, Y_data = shuffle(X_data, Y_data, random_state = 0)
	X_data = X_data/255
	return X_data, Y_data

#Function do the train-val split
def get_train_val_data(X_data, Y_data, ratio):
	X_train, X_val, Y_train, Y_val = train_test_split(X_data, Y_data, test_size = ratio)
	return X_train, X_val, Y_train, Y_val

#Function to sample k random centers
def get_random_centers(X_train, N, K):
	#Convert list of centers into numpy array for faster calculations
	centers = np.array(X_train[random.sample(range(N), K)])	#Probability that we select a center twice is very low.
	return centers

#Function for k-means++ sampling
def kmeansPP(X_train, N, K):
	centers = []
	#Selecting the first point at random
	next_center_idx = random.randint(0, N-1)
	centers.append(X_train[next_center_idx])

	#We will find the next K-1 centers
	while(len(centers) != K):
		np_centers = np.array(centers)
		X_squared = np.sum(X_train*X_train, axis = 1)
		centers_squared = np.sum(np_centers*np_centers, axis = 1)
		distance_matrix = np.asmatrix(X_squared).T + np.asmatrix(centers_squared) - 2*(np.dot(X_train, np_centers.T))
		distance_matrix = np.maximum(distance_matrix, 0)	#Distance of a point to itself sometimes become negative because of precision issues
		probability_list = np.array(np.min(distance_matrix, axis = 1).T)[0]
		total_cost = np.sum(probability_list)
		probability_list = probability_list/total_cost
		next_center_idx = np.random.choice(N, p = probability_list)
		centers.append(X_train[next_center_idx])

	#Convert list of centers into numpy array for faster calculations
	centers = np.array(centers)
	return centers

#Function for clustering the data
def get_clustering(X_train, N, K, centers):
	current_clustering = []
	cur_cost = 0	#Cost of the current clustering

	#Centers is KxM dimensional, X_train is NxM dimensional
	#We will get the clustering by vectorising all operations for much much faster processing
	X_squared = np.sum(X_train*X_train, axis = 1)
	centers_squared = np.sum(centers*centers, axis = 1)
	distance_matrix = np.asmatrix(X_squared).T + np.asmatrix(centers_squared) - 2*(np.dot(X_train, centers.T))
	distance_matrix = np.maximum(distance_matrix, 0)	#Distance of a point to itself sometimes become negative because of precision issues

	current_clustering = np.array(np.argmin(distance_matrix, axis = 1).T)[0]
	
	cur_cost = np.sum(np.min(distance_matrix, axis = 1))

	iteration = 0	#Number of iterations
	
	#Now we will cluster till the convergence criteria has been met
	while(True):
		#Define a KxN matrix that will be used to directly obtain new centers
		multiplier = np.zeros((K, N))
		multiplier[current_clustering, np.arange(N)] = 1.0	#Assign ones at useful positions

		#Number of points in each cluster
		num_points = np.sum(multiplier, axis = 1)
		
		#Obtain the new centers
		new_centers = np.dot(multiplier, X_train)
		for i in range(K):
			new_centers[i] = new_centers[i]/num_points[i]
		centers = new_centers

		#We will remember the previous clustering, as this can be used for convergence
		previous_clustering = current_clustering 
		
		#We will remember the previous cost, as this can be used for convergence
		prev_cost = cur_cost

		#To get the new clustering
		X_squared = np.sum(X_train*X_train, axis = 1)
		centers_squared = np.sum(centers*centers, axis = 1)
		distance_matrix = np.asmatrix(X_squared).T + np.asmatrix(centers_squared) - 2*(np.dot(X_train, centers.T))
		distance_matrix = np.maximum(distance_matrix, 0)	#Distance of a point to itself sometimes become negative because of precision issues

		current_clustering = np.array(np.argmin(distance_matrix, axis = 1).T)[0]

		cur_cost = np.sum(np.min(distance_matrix, axis = 1))
		print(prev_cost -cur_cost)
		iteration += 1
		if(np.linalg.norm(current_clustering-previous_clustering) == 0):
			break

	return current_clustering, centers

#Function to find out the label represented by each cluster
def get_cluster_labels(current_clustering, Y_train, N, K, num_labels):
	#Some structures defined for this purpose
	cluster_freq = []
	for i in range(K):
		cluster_freq.append([0]*num_labels)

	#Find the frequency of all label in every cluster
	for i in range(N):
		cluster_freq[current_clustering[i]][Y_train[i]] += 1

	#Now find the label of the cluster
	cluster_labels = []
	addup = 0
	for i in range(K):
		temp = max(cluster_freq[i])
		addup += temp
		cluster_label = [i for i, j in enumerate(cluster_freq[i]) if j == temp]
		cluster_labels.append(cluster_label[0])

	print("Purity = ", addup/N)

	cluster_labels = np.array(cluster_labels)
	return cluster_labels

#Function to make final predictions
def make_predictions(X_test, cluster_labels, centers):
	#Obtain clusters of test data
	X_squared = np.sum(X_test*X_test, axis = 1)
	centers_squared = np.sum(centers*centers, axis = 1)
	distance_matrix = np.asmatrix(X_squared).T + np.asmatrix(centers_squared) - 2*(np.dot(X_test, centers.T))
	distance_matrix = np.maximum(distance_matrix, 0)	#Distance of a point to itself sometimes become negative because of precision issues

	current_clustering = np.array(np.argmin(distance_matrix, axis = 1).T)[0]

	Y_pred = cluster_labels[current_clustering]
	return Y_pred

def main():
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]
	X_train, Y_train = get_data(train_path)

	N = X_train.shape[0]	#Number of training examples
	num_labels = np.max(Y_train)+1	#Number of labels
	K = 2300	#Number of clusters to form. This is a hyperparameter

	centers = get_random_centers(X_train, N, K)
	final_clustering, centers = get_clustering(X_train, N, K, centers)
	cluster_labels = get_cluster_labels(final_clustering, Y_train, N, K, num_labels)

	del X_train
	del Y_train

	X_test = np.load(test_path)
	X_test = X_test/255

	Y_pred = make_predictions(X_test, cluster_labels, centers)

	out_file = open(out_path, 'w')
	for i in range(len(Y_pred)):
		out_file.write(str(Y_pred[i])+'\n')
	out_file.close();

if __name__ == '__main__':
	main()