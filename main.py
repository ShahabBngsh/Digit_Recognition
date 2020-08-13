import matplotlib.pyplot as plt #for plotting
#have to install this library for first time
# py -3.7 pip install idx2numpy
import idx2numpy 
#import pandas as pd
import numpy as np # for numerical calculation with ndarray

# local files
import nnPy

# load the data(in idx format) to numpy's ndarray
X3d = idx2numpy.convert_from_file (
   "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-images-idx3-ubyte"
)
y = idx2numpy.convert_from_file (
   "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-labels-idx1-ubyte"
)


print(y.size)
# #plot the image using pyplot
# print(y[0])
# plt.imshow(X3d[0])
# plt.show()

#reshape 3d matrix to 2d
# e.g: [10, 3, 3] --> [10, 9]
X = X3d.reshape(len(X3d), -1)

h = nnPy.Feedforward(X)

#encoding each row of 'y' in logical array
one_hot = np.zeros((y.size, y.max()+1))
rows = np.arange(y.size)
one_hot[rows, y] = 1

# f1 = np.matmul(y, (np.log(h)).transpose())
# print(one_hot.shape, h.shape)








