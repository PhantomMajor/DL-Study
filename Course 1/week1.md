# PYTHON BASICS WITH NUMPY

**General Rules**:
* We will use python 3
* Avoid using explict for-loops

---

## 1 - Building basic functions with numpy ##

Numpy is the main package for scientific computing in Python.Key numpy functions are:
* np.exp
* np.log
* np.reshape.

### 1.1 - sigmoid function ###


    sigmoid (z) = 1 / ( 1 + exp(-z) )


To refer to a function belonging to a specific package you could call it using package_name.function().

#### The sigmoid function:

```py
import numpy as np

def sigmoid (x):
    """
    Compute the sigmoid of x

    Arguments:
    x -- A scalar or a numpy array of anu size

    Return:
    s -- sigmoid (x)
    """

    s = 1 / ( 1 + np.exp(-x) )

    return s
```

### 1.2 - Sigmoid gradient

        sigma'(x) = sigma(x) * (1 - sigma(x))

```py
def sigmoid_derivative (x):
    """
    Computes the derivative of the sigmoid with respect to the input x

    Arguments:
    x -- a scalar or a numpy array

    Return:
    ds -- computed gradient
    """

    ds = sigmoid(x) * ( 1 - sigmoid(x) )

    return ds
```

### 1.3 - Reshaping arrays ###

Two common numpy functions used in deep learning are [np.shape](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ndarray.shape.html) and [np.reshape()](https://docs.scipy.org/doc/numpy/reference/generated/numpy.reshape.html).

- X.shape is used to get the shape (dimension) of a matrix/vector X.
- X.reshape(...) is used to reshape X into some other dimension.

```py
def image2vector (image):
    """
    Takes an input of shape (length, height, 3) and returns a vector of shape (length*height*3, 1).

    Arguments:
    image -- a numpy array of shape (length, height, depth)

    Returns:
    v -- a vector of shape (length*height*depth, 1)
    """

    v = np.reshape( ( image.shape[0] * image.shape[1] * image.shape[2], 1 ) )

    return v
```

### 1.4 - Normalizing rows

Another common technique we use in Machine Learning and Deep Learning is to normalize our data. It often leads to a better performance because gradient descent converges faster after normalization.

Here, by normalization we mean dividing each row vector of x by its norm.

```py
norm(x) = np.linalg.norm(x, axis = 1, keepdims = True)

# it is the square root of the sum of squares of elements row-wise
```

```py
def normalizeRows(x):
    """
    Implement a function that normalizes each row of the matrix x (to have unit length).

    Argument:
    x -- A numpy matrix of shape (n, m)

    Returns:
    x -- The normalized (by row) numpy matrix.
    """

    x_norm = np.linalg.norm(x, ord = 2, axis = 1, keepdims = True)

    x = x/x_norm

    return x
```

---

## 2-Vectorization ##

**Note** that `np.dot()` performs a matrix-matrix or matrix-vector multiplication. This is different from `np.multiply()` and the `*` operator  which performs an element-wise multiplication.

### 2.1 Implement the L1 and L2 loss functions

The loss is used to evaluate the performance of your model. The bigger your loss is, the more different your predictions are from the true values. In deep learning, you use optimization algorithms like Gradient Descent to train your model and to minimize the cost.

L1 loss is just the sum of absolute differences of the predictions and the actual labels. i.e.

    sum(|yhat - y|)

```py
def L1(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of the L1 loss function defined above
    """

    loss = np.sum(abs(y - yhat), axis = 0)

    return loss
```

L2 loss is defined as:

    sum( ( yhat-y )^2 )

```py
def L2(yhat, y):
    """
    Arguments:
    yhat -- vector of size m (predicted labels)
    y -- vector of size m (true labels)

    Returns:
    loss -- the value of L2 loss function defined above
    """

    loss = np.sum( np.dot(yhat-y, yhat-y), axis=0 )

    return loss
```
