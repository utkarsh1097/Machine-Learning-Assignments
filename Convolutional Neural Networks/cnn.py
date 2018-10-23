import pandas as pd
import numpy as np
import tensorflow as tf
import sys

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Activation, Flatten, Dense, Dropout, BatchNormalization


def one_hot_encoder(Y):
	depth = np.max(Y) + 1
	N = Y.shape[0]
	new_Y = np.zeros((N, depth))
	new_Y[np.arange(N), Y] = 1
	return new_Y

def load_train(data_path):
	data_df = pd.read_csv(data_path, header = None)

	X_data = data_df.iloc[:, 1:].values
	Y_data = data_df.iloc[:, 0].values

	X_data = X_data/255

	X_data = X_data.reshape((-1, 32, 32, 1))

	Y_data = one_hot_encoder(Y_data)

	del data_df

	return X_data, Y_data

def load_test(data_path):
	data_df = pd.read_csv(data_path, header = None)

	X_data = data_df.iloc[:, 1:].values

	X_data = X_data/255

	X_data = X_data.reshape((-1, 32, 32, 1))

	return X_data

def cnn_model():
	model = Sequential()

	model.add(Conv2D(filters=20, kernel_size = 3, strides = 1, activation='relu', padding='same', input_shape=(32, 32, 1)))
	model.add(Conv2D(filters=20, kernel_size = 3, strides = 1, activation='relu', padding='same'))
	model.add(Conv2D(filters=20, kernel_size = 3, strides = 1, activation='relu', padding='same',))
	model.add(Dropout(0.2))

	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(Conv2D(filters=50, kernel_size = 3, strides = 1, activation='relu', padding='same'))
	model.add(Conv2D(filters=50, kernel_size = 3, strides = 1, activation='relu', padding='same'))
	model.add(Conv2D(filters=50, kernel_size = 3, strides = 1, activation='relu', padding='same'))
	model.add(Dropout(0.5))

	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(Conv2D(filters=100, kernel_size = 3, strides = 1, activation='relu', padding='same'))
	model.add(Conv2D(filters=100, kernel_size = 3, strides = 1, activation='relu', padding='same'))
	model.add(Conv2D(filters=100, kernel_size = 3, strides = 1, activation='relu', padding='same'))
	model.add(Dropout(0.5))

	model.add(MaxPool2D(pool_size=(2, 2), strides=(2, 2)))

	model.add(Flatten())
	model.add(Dense(500, activation='relu'))
	model.add(Dropout(0.5))
	
	model.add(Dense(46, activation='softmax'))

	return model

def main():
	#Command line arguments
	train_path = sys.argv[1]
	test_path = sys.argv[2]
	out_path = sys.argv[3]

	#Load in train and test datasets
	X_train, Y_train = load_train(train_path)
	X_test = load_test(test_path)

	model = cnn_model()

	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

	batch_size = 128
	num_epochs = 50

	model.fit(X_train, Y_train, batch_size=batch_size, epochs=num_epochs, verbose=1)
	
	prediction = model.predict(X_test, batch_size=batch_size)
	
	labels = np.argmax(prediction, axis = 1)
	out_file = open(out_path, 'w')

	for i in range(len(labels)):
		out_file.write(str(labels[i])+"\n")
	
	out_file.close()

if __name__ == '__main__':
	main()