import matplotlib.pyplot as plt #for plotting
#have to install first time
# py -3.7 pip install idx2numpy
import idx2numpy 

#import pandas as pd
import numpy as np

# load the data(in idx format) to numpy's ndarray
X3d = idx2numpy.convert_from_file("E:\Courses\LocalRepo\Digit_Recogniser\\t10k-images-idx3-ubyte")
y   = idx2numpy.convert_from_file("E:\Courses\LocalRepo\Digit_Recogniser\\t10k-labels-idx1-ubyte")


# # print(len(X3d))
# #plot the image using pyplot
# print(y[0])
# plt.imshow(X3d[0])
# plt.show()

#reshape 3d matrix to 2d
# e.g: [10, 3, 3] --> [10, 9]
X = X3d.reshape(len(X3d), -1)
# we've to add bias unit as always
X = np.c_[np.ones(len(X)), X]

#random weights for our feedforward propogation
# 25 neurons in hidden layer + bias unit
theta1 = np.random.rand(25, X.shape[1])
# there will be 10 classes
theta2 = np.random.rand(10, 26)
print(theta1.shape)












