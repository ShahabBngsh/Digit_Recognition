import matplotlib.pyplot as plt #for plotting
#have to install first time
# py -3.7 pip install idx2numpy
import idx2numpy 

#import pandas as pd
#import numpy as np

#load the data(in idx format) to numpy's ndarray
arr = idx2numpy.convert_from_file("E:\Courses\LocalRepo\Digit_Recogniser\\t10k-images-idx3-ubyte")


print(len(arr))
#plot the image using pyplot
"""plt.imshow(arr[12])
plt.show()
"""
















