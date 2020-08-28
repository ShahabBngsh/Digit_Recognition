# %%
# Title
import numpy as np
import idx2numpy
import matplotlib.pyplot as plt
import keyboard
import time
#local files
import nnPy

#Train Test split
X3d = idx2numpy.convert_from_file (
  # "E:\Courses\LocalRepo\Digit_Recogniser\\train-images-idx3-ubyte"
	"E:\Courses\LocalRepo\Digit_Recogniser\\t10k-images-idx3-ubyte"

)
y = idx2numpy.convert_from_file (
  # "E:\Courses\LocalRepo\Digit_Recogniser\\train-labels-idx1-ubyte"
  "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-labels-idx1-ubyte"
)

X3d_test = idx2numpy.convert_from_file (
  "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-images-idx3-ubyte"
)
y_test = idx2numpy.convert_from_file (
  "E:\Courses\LocalRepo\Digit_Recogniser\\t10k-labels-idx1-ubyte"
)


#reshape 3d matrix to 2d
# e.g: [10, 3, 3] --> [10, 9]
X_u = X3d.reshape(len(X3d), -1)
X = X_u/255

X_u_test = X3d_test.reshape(len(X3d_test), -1)
# each pixel is in range of 0-255 --> 0-1
X_test = X_u_test/255 #data normalisation

#random initial weights
W = nnPy.InitRandWeights(X_test.shape[1], 25, y_test.max()+1)
accuracy = []
loss = []
#learning rate
alpha = 8
i=0

#main loop
print("Training ... ", "It'll take a while", "Set back and relax ;) ", sep='\n')
while(i < 60): 
   
	W, AllCost = nnPy.NN_Model(X, y, W, 10, alpha)
	prd = nnPy.Predict(X, W)
 
	accuracy.append(np.mean(prd==y))
	loss.append(np.mean(AllCost))
 
	print("Accuracy: ", accuracy[-1])
	print("Loss:     ", loss[-1])
 
	if accuracy[-1] > np.float(0.95):
		break
	elif accuracy[-1] > np.float(0.90):
		alpha = 2
	elif accuracy[-1] > np.float(0.80):
		alpha = 2.5
	elif accuracy[-1] > np.float(0.75):
		alpha = 3
	elif accuracy[-1] > np.float(0.65):
		alpha = 4
	elif accuracy[-1] >= np.float(0.55):
		alpha = 5
  
	i+=1
print("I'm done")
#plot accuracy and loss values with each iteration
# fig = plt.figure()
# for i in range(1, len(loss)):
#   plt.plot([i-1, i], [accuracy[i-1], accuracy[i]], color='blue',
#            label='Accuracy'
#           )
#   plt.plot([i-1, i], [loss[i-1], loss[i]], color='red', label='Loss')
#   plt.pause(0.1)
#   plt.legend(['Accuracy', 'Loss'])
# plt.show()

# pick random pics
# print("Press 'esc' to exit", "'enter' to continue", sep='\n')
# while (not keyboard.is_pressed('esc')):
   
#    if keyboard.is_pressed('esc'):
#       break
#    elif keyboard.is_pressed('enter'):
#       print("Press 'esc' to exit", "'enter' to continue", sep='\n')
#       index = np.random.randint(0, y.shape[0])
#       plt.imshow(X3d[index])
#       plt.title("True Label: " + str(y[index]))
#       plt.xlabel("Predicted Label: " + str(prd[index]))
#       time.sleep(0.5)
#       plt.show()
  
# %%
# confusion matrix
prd_test = nnPy.Predict(X_test, W)
classes = np.max(y_test) + 1
confMat = np.zeros([classes, classes], dtype=int)
for i in range(len(y_test)):
  confMat[y_test[i]][prd_test[i]] += 1
confMat = np.round(confMat/confMat.sum(axis=1), decimals=3)


# %%
# plot confusion matrix
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

# %%
