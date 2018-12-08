# Planar data classification with one hidden layer

We will build a neural network with one hidden layer.

**We will learn how to:**

-   Implement a 2-class classification neural network with a single hidden layer
-   Use units with a non-linear activation function, such as tanh
-   Compute the cross entropy loss
-   Implement forward and backward propagation

## 1-Packages

-   [numpy](www.numpy.org) is the fundamental package for scientific computing.
-   [sklearn](http://scikit-learn.org/stable/) provides simple tools for data mining and analysis.
-   [matplotlib](http://matplotlib.org) is a library for plotting graphs in Python.
-   planar_utils provide various useful functions used in this assignment like sigmoid

```py
import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets
```

## 2-Dataset

The following code will load a "flower" 2-class dataset into the variables X and Y

```py
X, Y = load_planar_dataset()
```

We can visualise the dataset using matplotlib. The data looks like a flower with some red and some blue points. Red points have label y=0 and blue have label y=1

```py
plt.scatter(X[0, :], X[1, :], c=Y, s=40, cmap=plt.cm.Spectral)
```

We have

-   A numpy matrix X containing the features (x1 and x2)
-   A numpy vector Y containing the labels (red:0, blue:1)

To get a better sense of what our data is like:

```py
# This can be done in the following two formats:
shape_X = X.shape
shape_Y = np.shape(Y)
m = X.shape[1]  # training set size

print ("The shape of X is:" + str(shape_X))
print ("The shape of Y is:" + str(shape_Y))
print ("I have m = %d number of training examples!" % (m) )
```

## 3-Neural Network Model

** Here is our Model **

<img src="classification_kiank.png" style="width:600px;height=300px;">

If probability is &lt; .5, then prediction is 0

The loss function we use will be the same as the one we used in logistic regression.

** The General Method to Build a Neural Network **
1\. Define the Neural Network structure. (# of input units, # of hidden layers, etc)
2\. Randomly initialize the model's parameters.
3\. Loop:
    \- Implement forward propodation.
    \- Compute loss.
    \- Implement backward propogation to get the gradients.
    \- Update parameters (gradient descent)

### 3.1-Defining the Neural Network Structure

```py
def layer_sizes (X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of size (output size, number of examples)

    Returns:
    n_x -- size of input layer
    n_h -- size of hidden layer
    n_y -- size of output layer
    """

    n_x = X.shape[0]
    n_h = 4  # we are defining this, hence it is hardcoded!
    n_y = Y.shape[0]

    return (n_x, n_h, n_y)
```

## 3.2-Initialize the Model's Parameters

```py
def initialize_parameters (n_x, n_h, n_y):
    """
    Arguments:
    n_x -- input layer size
    n_h -- hidden layer size
    n_y -- output layer size

    Returns:
    parameters -- python dictionary containing the initialized parameters:
        W1 -- weight matrix of shape (n_h, n_x)
        b1 -- bias vector of shape (n_h, 1)
        W2 -- weight matrix of shape (n_y, n_h)
        b2 -- bias vector of shape (n_y, 1)
    """

    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros(n_h, 1)
    W2 = np.rendom.randn(n_y, n_h) * 0.01
    b2 = np.zeros(n_y, 1)

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
```

## 3.3-The Loop

```py
def forward_propagation (X, parameters):
    """
    Arguments:
    X -- input data of size (n_x, m)
    parameters -- dictionary of parameters (output of initialization function)

    Returns:
    A2 -- sigmoid output of the second activation.
    cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
    """

    # Retreving the parameters
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # implementing forward propagation
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.sigmoid(Z2)

    assert ( A2.shape == (1, X.shape[1]) )

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}

    return A2, cache
```

Now we claculate the cross-entropy loss

```py
def compute_cost (A2, Y, parameters):
    """
    Arguments:
    A2 -- The sigmoid output of the seconf activation of the shape(1, number of examples)
    Y -- "true" label vector of shape (1, number of examples)
    parameters -- python dictionary containing the parameters W1, b1, W2 and b2

    Returns:
    cost -- cross entropy cost given by:
        J = (-1/m) * horizontal_sum[ { yi*log(ai) } + { (1-yi)*log(1-ai) } ]

    """

    m = Y.shape[1]  # number of training examples

    # Compute the cross entropy loss
    log_probs = ( Y * np.log(A2) ) + ( ( 1-Y ) * ( np.log(1 - A2) ) )
    cost = (-1/m) * np.sum(log_probs)

    cost = np.squeeze(cost)  # makes sure the dimensions of cost is what we expect
                             # Eg.  Turns [[17]] into 17

    assert( isinstance(cost, float) )

    return cost
```

Using the cache collected during forward propagation, we can implement backward propagation.

**Instructions**:
Backpropagation is usually the hardest (most mathematical) part in deep learning. Use the six equations on the right of this slide, since we are building a vectorized implementation.  

<img src="grad_summary.png" style="width:600px;height:300px;">

```py
def backward_propagation (parameters, cache, X, Y):
    """
    Arguments:
    parameters -- python dictionary containing the parameters W1, b1, W2 and b2
    cache -- python dictionary containing Z1, A1, Z2 and A2
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)

    Returns:
    grads -- python dictionary containing gradients with respect ot different parameters
    """

    m = X.shape[1]

    # retrive W1 and W2 from "parameters"
    W1 = parameters["W1"]
    W2 = parameters["W2"]

    #retrive A1 and A2 form "cache"
    A1 = cache["A1"]
    A2 = cache["A2"]

    # Backward propagation. Calculate dW1, db1, dW2 and db2
    dZ2 = A2 - Y  # formula has been derived for sigmoid
    dW2 = (1/m) * np.dot( dZ2, A1.T )
    db2 = (1/m) * np.sum( dZ2, axis=1, keepdims=True )

    dZ1 = np.dot( W2.T, dZ2 ) * ( 1-np.power(A1, 2) )  # formula of derivative of tanh
    dW1 = (1/m) * np.dot( dZ1, X.T )
    db1 = (1/m) * np.sum( dZ1, axis=1, keepdims=True )

    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2", db2}

    return grads
```

Now we just have to implement gradient descent to learn teh parametrs.

**Illustration**: The gradient descent algorithm with a good learning rate (converging) and a bad learning rate (diverging). Images courtesy of Adam Harley.

<img src="sgd.gif" style="width:400;height:400;"> <img src="sgd_bad.gif" style="width:400;height:400;">

```py
def update_parameters (parameters, grads, learning_rate = 1.2):
    """
    Arguments:
    parameters -- python dictionary containing the parameters W1, b1, W2 and b2
    grads -- python dictionary containing the gradients dW1, db1, dW2 and db2

    Returns:
    parameters -- python dictionary containg updated parameters
    """

    # Retrive each parameter
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrive each gradient
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update the parameters
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * bd2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}

    return parameters
```

## 3.4-Integrate part 3.1, 3.2 and 3.3 in one model

```py
def nn_model (X, Y, n_h, num_iterations=10000, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (2, number of examples)
    Y -- labels of shape (1, number of examples)
    n_h -- size of the hidden layer
    num_iterations -- Number of iterations in gradient descent loop
    print_cost -- if True, print the cost every 1000 iterations

    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """

    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize the parameters and retrive them.
    parameters = initialize_parameters (n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # loop for gradient descent
    for i in range(0, num_iterations):

        # Forward propagation
        A2, cache = forward_propagation(X, parameters)

        # cost calculation
        cost = compute_cost(A2, Y, parameters)

        # Backward propagation
        grads = backward_propagation(parameters, cache, X, Y)

        # update parameters
        parameters = update_parameters(parameters, grads, learning_rate= 1.2)

        # print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print("Cost after iteration %i: %f" % (i, cost) )

        return parameters

```

## 3.5-Predictions ##

```py
def predict (parameters, X):
    """
    Arguments:
    parameters -- python dictionary containing your parameters
    X -- input data of size (n_x, m)

    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """

    # Computes probabilities using forward propagation, and classifies to 0/1 using 0.5 as the threshold.

    A2, cache = forward_propagation(X, parameters)
    # will predict 1 if probability is greater than .5, else 0
    predictions = np.where(A2>.5, 1, 0)

    return predictions

```
