import numpy as np
import matplotlib.pyplot as plt

def InitRandWeights(in_f, n_h, n_o):
   """Initialise random weights with mean=0, variance = 1

   Args:
      in_f (int): # of feaures in Input layer
      n_h  (int): # of neurons in hidden layer
      n_o  (int): # of neurons in output layer

   Returns:
      dictionary: w1: n_h * in_f + 1, w2: n_o * n_h + 1
   """

   w1 = np.random.randn(n_h, in_f + 1)*2*0.1 - 0.1
   w2 = np.random.randn(n_o,  n_h + 1)*2*0.1 - 0.1

   Weights = {
		"w1": w1,
		"w2": w2
	}
   return Weights

def ForwardPass(X, W):
  """Perform a single forward pass

  Args:
    X (ndarray): in_f by m array,
    W (dictionary): should contain w1 and w2
       
  Returns:
    ZAs (dictionary): a1, z2, a2, z3, a3(our predicted values)
  """
  # number of training examples
  m = X.shape[1]
  
  A1 = np.r_[np.ones([1, m]), X]
  Z2 = np.matmul(W["w1"], A1)
  A2 = np.r_[np.ones([1, m]), Activation(Z2)]
  Z3 = np.matmul(W["w2"], A2)
  A3 = Activation(Z3)
	
  ZAs = {
		"a1": A1,
		"z2": Z2,
		"a2": A2,
		"z3": Z3,
		"a3": A3
	}
  return ZAs

def Activation(Z):
  """sigmoid activation function

  Args:
      Z (ndarry): can be vector or matrix

  Returns:
      ndarray: g(Z)
  """
  return 1.0/(1.0 + np.exp(-Z))

def ActGrad(Z):
  """sigmoid gradient g'(z)

  Args:
      Z (ndarray): can be vector or matrix

  Returns:
      ndarray: g'(z)
  """
  return Z*(1-Z)

def Cost(y, yhat):
  """Calculate difference b/w true labels and predicted labels
     using cross entropy

  Args:
      y (vector): m by 1 vector of true labels
      yhat (vector): 1 by m vector of predicted labels

  Returns:
      J (double): Loss value
  """
  
  m = y.shape[0]
  yhat = yhat.T
  loss = -(y*np.log(yhat)+(1-y)*np.log(1-yhat))
  return np.sum(loss)/m

def EncodeY(y):
  """Encode y values using one_hot encoding

  Args:
      y (vector): m by 1

  Returns:
      Y (ndarray): m by n_o, where n_o is # of neurons in output layer
  """
  one_hot = np.zeros([y.shape[0], y.max()+1])
  rows = np.arange(y.shape[0])
  one_hot[rows, y.T] = 1
  return one_hot

def BackPass(Y, ZAs, W):
  """Perform 1 backpropogation step

  Args:
      Y (matrix): one_hot encoded values of true labels
      ZAs (dictionary): return values from forward pass
      W (dictionary): Weights containing w1 and w2

  Returns:
      dicitonary: gradients of Weights w1 and w2
  """
  m = Y.shape[0]
  sigma3 = ZAs["a3"].T - Y
  sigma2 = np.matmul(sigma3, W["w2"]).T*ActGrad(ZAs["a2"])
  sigma2 = np.delete(sigma2, 0, axis=0)
  delta2 = np.matmul(ZAs["a2"], sigma3)/m
  delta1 = np.matmul(sigma2, ZAs["a1"].T)/m
  
  retdict = {
     			"dw1": delta1,
          "dw2": delta2
  }
  return retdict

def Optimize(W, dW, alpha):
  """Gradient descent: follow the steepest slope

  Args:
      W  (dictionary): {w1,   w2}
      dW (dictionary): {dw1, dw2}
      alpha (double): hyperparameter learning rate

  Returns:
      dictionary: new Weights
  """
  W["w1"] = W["w1"] - alpha*dW["dw1"]
  W["w2"] = W["w2"] - alpha*dW["dw2"].T
	
  return W

def NN_Model(x, y, W, numIters, alpha):
  """Perform forward pass, calculate errors, perform back prop
  and then perform gradient descent

  Args:
      x (matrix): m by in_f
      y (vector): m by 1 vector of true labels
      W (dictionary): Weights {w1, w2}
      numIters (int): number of iterations that needs to be performed
      alpha (double): learning rate

  Returns:
      CSVs: new Weights and vector of errors
  """
  x = x.T
  Y = EncodeY(y)
  AllCost = np.zeros([numIters, 1])
  print("...")
  for i in range(numIters):
    ZAs = ForwardPass(x, W)
    J = Cost(Y, ZAs["a3"])
    dWs = BackPass(Y, ZAs, W)
    AllCost[i] = J
    W = Optimize(W, dWs, alpha)
  return W, AllCost

def Predict(x, W):
  """Perform prediction, given optimized weights

  Args:
      x (matrix): m by in_f
      W (dictionary): Weights {w1, w2}

  Returns:
      vector: m by 1 vector of predicted values
  """
  ZAs = ForwardPass(x.T, W)
  h = ZAs["a3"]
  h = np.argmax(h, axis=0)
  return h

def ConfusionMatrix(y_labels, pred_labels, classes):
  
  confMat = np.zeros([classes, classes], dtype=int)
  for i in range(len(y_labels)):
    confMat[y_labels[i]][pred_labels[i]] += 1
  confMat = np.round(confMat/confMat.sum(axis=1), decimals=3)
  return confMat

def PlotConfMatrix(confMat, classes):
  fig, ax = plt.subplots(figsize=(22, 18))
  ax.matshow(confMat, cmap='Blues')
  for (i, j), z in np.ndenumerate(confMat):
    ax.text(i, j, z, va='center', ha='center', color='gray', size=20)
  ticks = np.arange(classes)
  plt.xticks(ticks)
  plt.yticks(ticks)
  plt.title('Confusion Matrix', size=16)
  plt.xlabel('Predicted Labels', size=14)
  plt.ylabel('Actual Labels', size=14)
  plt.plot()