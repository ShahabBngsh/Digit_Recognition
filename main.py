import matplotlib.pyplot as plt #for plotting
#have to install first time
# py -3.7 pip install idx2numpy
import idx2numpy 
#import pandas as pd
import numpy as np

# local files
import nnPy

# load the data(in idx format) to numpy's ndarray
X3d = idx2numpy.convert_from_file (
   "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-images-idx3-ubyte"
)
y = idx2numpy.convert_from_file (
   "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-labels-idx1-ubyte"
)


# # print(len(X3d))
# #plot the image using pyplot
# print(y[0])
# plt.imshow(X3d[0])
# plt.show()

#reshape 3d matrix to 2d
# e.g: [10, 3, 3] --> [10, 9]
X = X3d.reshape(len(X3d), -1)

nnPy.Feedforward(X)









