# Digit_Recogniser

> Digit Recognising using Neural Net of an 28*28 pixel images by MNIST.

> Everything was built from scratch upto the activation function.
---

## Model Loss and Accuracy
- It is designed to show how far we're from the 'ideal' solution.
- Closer the `accuracy` value to `1`, the better it is.

![](https://i.imgur.com/gulSBtG.gif)
---

## Confusion Matrix
- It shows how many images are correctly identified
- Matrix is normalised so, values will be b/w [0, 1].
- Ideally principal diagonal values should be 1, other values should be 0

[![Confusion matrix image](https://i.imgur.com/mvnKrhA.jpg)]()
---

## Dependencies
- We assume that `Anaconda` is already installed.
- Anaconda comes with important libraries like `numpy` and `matplotlib`

> Update and install `PyPl's keyboard` library from terminal
```shell
py -version -m pip install keyboard
```
---

## Setup
- download MNIST dataset from <a href="https://www.kaggle.com/hojjatk/mnist-dataset"> Kaggle</a>.
- provide path to the dataset in main.py file
```python
#Train Test split
X3d = idx2numpy.convert_from_file (
  "E:\Path\to\trainingset images\\train-images-idx3-ubyte"
)
y = idx2numpy.convert_from_file (
  "E:\Path\to\trainingset labels\\train-labels-idx1-ubyte"
)

X3d_test = idx2numpy.convert_from_file (
  "E:\Path\to\testset images\\t10k-images-idx3-ubyte"
)
y_test = idx2numpy.convert_from_file (
  "E:\Path\to\testset labels\\t10k-labels-idx1-ubyte"
)
```
---

## Thanks
- Thanks to the `google` and `stackoverflow`. They were always there for my 101 questions ;)
- Thank you for reading to the end :)