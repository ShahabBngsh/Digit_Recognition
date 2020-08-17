# %%
import matplotlib.pyplot as plt #for plotting
#have to install this library for first time
# py -3.7 pip install idx2numpy
import idx2numpy 
#import pandas as pd
import numpy as np # for numerical calculation with ndarray

# local files
# import nnPy

# %%
def ReLU(z):
   """ReLU function g(z) for NN

   Args:
       z (double ndarray): numpy's ndarray

   Returns:
       double ndarray: numpy's ndarray
   """
   # return np.maximum(0, z)
   return 1.0/(1.0 + np.exp(-z))

def ReLUGrad(z):
   # z = np.maximum(z, 0)
   # z = np.minimum(1, z)
   # return z
   return ReLU(z)*(1.0 - ReLU(z))

def InitParam(features, hiddenSize, outSize):
   #random weights for our feedforward propogation
   # 25 neurons in hidden layer, 784(28*28 pixels) features + 1 bias unit
   theta1 = np.random.randn(hiddenSize, features + 1)*0.01
   # there will be 10 classes
   theta2 = np.random.randn(outSize, hiddenSize + 1)
   b1 = np.zeros([hiddenSize, 1])
   b2 = np.zeros([outSize, 1])
   
   params = {
      "W1": theta1,
      "W2": theta2,
      "b1": b1,
      "b2": b2
   }
   return params

def Hypothesis(X, theta1, theta2):
   """
   Feedforward propogation with single hidden layer

   Args:
      X (ndarray): numpy's ndarray.
      You don't have to add bias unit.
      
   Returns:
      params: dictionary containing activation values of forward pass
   
   """

   #feedforward propogation
   # we've to add bias unit as always
   a0 = np.c_[np.ones(len(X)), X]
   z1 = np.matmul(a0, theta1.transpose()) # a0*theta1'
   a1 = ReLU(z1)
   a1 = np.c_[np.ones(a0.shape[0]), a1]
   z2 = np.matmul(a1, theta2.transpose())
   a2 = ReLU(z2)
   
   params = {
      "a0": a0,
      "a1": a1,
      "a2": a2,
      "z1": z1,
      "z2": z2
   }
   return params

def Cost(one_hot, h):
   m = len(one_hot)
  
   f1 = -(np.multiply(   one_hot,  np.log(h)))
   f2 = -(np.multiply((1-one_hot), np.log(1-h)))
   
   # print(f1.shape)
   # print(f2.shape)
   
   J = (1/m)*np.sum(np.sum(f1 + f2))
   return J

def BackProp(X, Y, W, params):
   
   m = len(X)
   
   dZ2 = params["a2"] - Y
  
   # print("dZ2 ", dZ2.shape)
   # print("a1", params["a1"].shape)
   dW2 = (1/m) * np.dot(dZ2.T, params["a1"])
   db2 = (1/m) * np.sum(dZ2.T, axis=1, keepdims=True)
   
   # print(params["z1"].shape)

   z1 = np.c_[np.ones(len(X)), params["z1"]]
   dZ1 = np.dot(dZ2, W["W2"]) * (ReLUGrad(z1))
   dZ1 = np.delete(dZ1, 0, axis=1)
   # print("Z1 ", z1.shape, "dZ1 ", dZ1.shape, "a0", params["a0"].shape)
   
   
   dW1 = (1/m)*np.matmul(dZ1.T, params["a0"])
   db1 = (1/m)*np.sum(dZ1.T, axis=1, keepdims=True)
   
   grad = {
      "db1": db1,
      "db2": db2,
      "dW1": dW1,
      "dW2": dW2
   }
   return grad

def UpdateParam(params, grads, alpha):
   
   W1 = params["W1"]
   b1 = params["b1"]
   W2 = params["W2"]
   b2 = params["b2"]
   
   dW1 = grads["dW1"]
   dW2 = grads["dW2"]
   db1 = grads["db1"]
   db2 = grads["db2"]
   
   W2 = W2 - alpha*dW2
   W1 = W1 - alpha*dW1
   b2 = b2 - alpha*db2
   b1 = b1 - alpha*db1
   
   
   params = {
      "W1": W1,
      "W2": W2,
      "b1": b1,
      "b2": b2
   }
   return params
   
def NN_Model(X, Y, numIters, alpha):
   
   params = InitParam(X.shape[1], 25, 10)
   print("Training ", end='')
   for i in range(0, numIters):
      
      cache = Hypothesis(X, params["W1"], params["W2"])
      
      J = Cost(Y, cache["a2"])
      
      W = {
         "W1": params["W1"],
         "W2": params["W2"]
      }
      grads = BackProp(X, Y, W, cache)
      
      params = UpdateParam(params, grads, alpha)
      print(".", end='')
   print(" ")
   return params
   
def Predict(X, params):
   
   cache = Hypothesis(X, params["W1"], params["W2"])
   h = cache["a2"]
   h = np.amax(h, axis=1)
   return h
   
   
# %%
# load the data(in idx format) to numpy's ndarray
X3d = idx2numpy.convert_from_file (
   "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-images-idx3-ubyte"
)
y = idx2numpy.convert_from_file (
   "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-labels-idx1-ubyte"
)


# %%
#reshape 3d matrix to 2d
# e.g: [10, 3, 3] --> [10, 9]
X = X3d.reshape(len(X3d), -1)
init = InitParam(X.shape[1], 25, 10)


Weights = {
   "W1": init["W1"],
   "W2": init["W2"]
}
W1 = init["W1"]
W2 = init["W2"]
   
m = len(X) # size of training set
dictff = Hypothesis(X, W1, W2)
h = dictff["a2"]

# %%
#encoding each row of 'y' in logical array
one_hot = np.zeros((y.size, y.max()+1))
rows = np.arange(y.size)
one_hot[rows, y] = 1

# %%

# compute cost J(0)
J = Cost(one_hot, h)
# print(one_hot.shape, h.shape)
grads = BackProp(X, one_hot, Weights,  dictff)

# %%
# print(J)
params = NN_Model(X, one_hot, 50, 0.01)
prd = Predict(X, params)
print(np.mean(prd))
# %%
