#!/usr/local/bin/python3

from pyTsetlinMachine.tm import MultiClassConvolutionalTsetlinMachine
import numpy as np 
from time import time

train_data = np.loadtxt("2DNoisyXORTrainingData.txt").astype(np.uint32)
X_train = train_data[:,0:-1].reshape(train_data.shape[0], 4, 4)
Y_train = train_data[:,-1]

test_data = np.loadtxt("2DNoisyXORTestData.txt").astype(np.uint32)
X_test = test_data[:,0:-1].reshape(test_data.shape[0], 4, 4)
Y_test = test_data[:,-1]

ctm = MultiClassConvolutionalTsetlinMachine(40, 60, 3.9, patch_dim=(2, 2))

average = 0.0
for i in range(100):
	start = time()
	ctm.fit(X_train, Y_train, epochs=5000)
	stop = time()

	average += ctm.evaluate(X_test, Y_test)
	print("%d: %.2f% %.1fs" % (i+1, 100*average/(i+1), stop-start))



