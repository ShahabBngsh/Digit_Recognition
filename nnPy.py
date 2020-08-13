import numpy as np

def Sigmoid(z):
   """sigmoid function g(z) for NN

   Args:
       z (double ndarray): numpy's ndarray

   Returns:
       double ndarray: numpy's ndarray
   """
   return 1.0/(1.0 + np.exp(-z))

def Feedforward(X):
   """Feedforward propogation with single hidden layer

   Args:
      X (ndarray): numpy's ndarray.
      You don't have to add bias unit.
      
   Returns:
      h(0): dimensions: [X.shape[0], 10]
   """
   
   #random weights for our feedforward propogation
   # 25 neurons in hidden layer, 784(28*28 pixels) features + 1 bias unit
   theta1 = np.random.rand(25, X.shape[1] + 1)
   # there will be 10 classes
   theta2 = np.random.rand(10, 26)

   #feedforward propogation
   # we've to add bias unit as always
   a1 = np.c_[np.ones(len(X)), X]
   z2 = np.matmul(a1, theta1.transpose()) # a1*theta1'
   a2 = Sigmoid(z2)
   a2 = np.c_[np.ones(a1.shape[0]), a2]
   z3 = np.matmul(a2, theta2.transpose())
   a3 = Sigmoid(z3)
   h  = a3
   
   return h

def Cost(X, y, h):  
   print("Homie")