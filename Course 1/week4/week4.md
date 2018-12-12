# Building A Deep Neural Net

**Notation**:
- Superscript[l] denotes a quantity associated with the l - th layer.
    - _Example_: `a ^ [L]` is the L - th layer activation. `W ^ [L]` and `b ^ [L]` are the L - th layer parameters.


- Superscript(i) denotes a quantity associated with the i - th example.
    - _Example_: `x ^ (i)` is the i - th training example.


- Lowerscript i denotes the i - th entry of a vector.
    - _Example_: `a ^ [l]_i` denotes the i - th entry of the l - th layer's activations).


## 1-Packages ##

```py

import numpy as np
import h5py
import matplotlib.pyplot as plt

```

## 2-Outline ##

1. Initialize the parameters for an L - layer Neural Network.
2. Implement the forward propagation module.
    - Complete a linear part of the layer's activation. This will result in ` Z ^ [l] `
    - Go from [LINEAR -> ACTIVATION] by using an activation function. This will result in ` A ^ [l] `
    - Stack together the[LINEAR -> ReLU] function L - 1 times and put a[LINEAR -> SIGMOID] at the end(since we are doing binary classification).
3. Compute the loss.
4. Implement the backward propagation.
    - Complete the LINEAR part of the layer's back prop. step.
    - Use the gradient of the ACTIVATE function to calculate the gradients.
    - Make a[LINEAR -> ACTIVATION] back function using the gradient of the sigmoid or the ReLU function.
    - Stack L - 1 [LINEAR -> RELU] functions and add a[LINEAR -> SIGMOID] function at the start.
5. update the parameters.


<img src="final outline.png" style="width:800px;height:500px;">
<caption> <center> **Figure 1**</center> </caption> <br>


## 3-Initialization ##

Write a helper function to initialize the parameters for L layers.

The initialization for a deeper L - layer neural network is more complicated because there are many more weight matrices and bias vectors.

Make sure that your dimensions match between each layer. Recall that ` n ^ [l] ` is the number of units in layer l. Thus for example if the size of our input X is (12288, 209)(with m = 209 examples) then:

<table style="width:100%">

    <tr>
        <td> </td>
        <td> **Shape of W ** </td>
        <td> **Shape of b ** </td>
        <td> **Activation ** </td>
        <td> **Shape of Activation ** </td>
    <tr>

    <tr>
        <td> **Layer 1 ** </td>
        <td> (n ^ [1], 12288) </td >
        <td> (n ^ [1], 1) </td>
        <td> Z ^ [1]=W ^ [1]  X + b ^ [1] </td>

        <td> (n ^ [1], 209) </td>
    <tr>

    <tr>
        <td> **Layer 2 ** </td >
        <td> (n ^ [2], n ^ [1]) </td>
        <td> (n ^ [2], 1) </td>
        <td> Z ^ [2]=W ^ [2] A ^ [1] + b ^ [2] </td>
        <td> (n ^ [2], 209) </td>
    <tr>

       <tr>
        <td> ... </td>
        <td> ... </td>
        <td> ... </td>
        <td> ... </td>
        <td> ... </td>
    <tr>

   <tr>
        <td> **Layer L - 1 ** </td>
        <td> (n ^ [L - 1], n ^ [L - 2]) </td>
        <td> (n ^ [L - 1], 1) </td>
        <td> Z ^ [L - 1]=W ^ [L - 1] A ^ [L - 2] + b ^ [L - 1] </td>
        <td> (n ^ [L - 1], 209) </td>
    <tr>


   <tr>
        <td> **Layer L ** </td>
        <td> (n ^ [L], n ^ [L - 1]) </td>
        <td> (n ^ [L], 1) </td>
        <td> Z ^ [L]=W ^ [L] A ^ [L - 1] + b ^ [L] </td>
        <td> (n ^ [L], 209) </td>
    <tr>


</table>


#### Implementing it:

```py

def initialize_parameters_deep(layer_dims):
    """
    Initialize the parameters for the L layer deep Neural Network

    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in our network

    Returns:
    parameters -- python dictionary containing parameters "W1", "b1", "W2", ... "WL", "bL"
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
    """

    parameters = {}  # create an empty dictionary of parameters
    L = len(layer_dims)  # the size of the Neural network

    for l in range(1, L):
        parameters["W" + str(l)] = np.random.randn( layer_dims[l], layer_dims[l-1] ) * 0.01
        parameters["b" +str(l)] = np.zeros( layer_dims[l], 1 ) * 0.01

        assert( parameters["W" + srt(l)].shape() == ( layer_dims[l], layer_dims[l-1] ) )
        assert( parameters["b" + str(l)].shape() == ( layer_dims[l], 1 ) )

    return parameters

```

## 4-Forward Propagation Module ##

### 4.1-Linear Forward ###

```py

def linear_forward(A, W, b):
    """
    Implement the linear part of the layer's forward propagation

    Arguments:
    A -- Activations from the previous layer (or the input data): (size of previous layer, number of examples)
    W -- Weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector: numpy array of shape (size of current layer, 1)

    Returns:
    Z -- input of the activation function (aka. pre-activation)
    cache -- python dictionary containing "A", "W" and "b"; stored for computing backward pass effectively.  
    """

    # calculating the pre-activataion
    Z = np.dot(W, A) + b

    assert( Z.shape() == (W.shape[0], A.shape[1]) )

    cache = (A, W, b)

    return Z, cache

```

## 4.2-Linear Activation Forward ##

We will use two activation functions:

- **Sigmoid**: $\sigma(Z) = \sigma(W A + b) = \frac{1}{ 1 + e^{-(W A + b)}}$. We have provided you with the `sigmoid` function. This function returns **two** items: the activation value "`a`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call: 
``` python
A, activation_cache = sigmoid(Z)
```

- **ReLU**: The mathematical formula for ReLu is $A = RELU(Z) = max(0, Z)$. We have provided you with the `relu` function. This function returns **two** items: the activation value "`A`" and a "`cache`" that contains "`Z`" (it's what we will feed in to the corresponding backward function). To use it you could just call:
``` python
A, activation_cache = relu(Z)
```
