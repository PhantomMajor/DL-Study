import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from lr_utils import load_dataset

# %matplotlib inline

# loading the data(cat/non-cat)
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


"""
# Example of a picture
index = 25
plt.imshow(train_set_x_orig[index])
print ("y = " + str(train_set_y[:, index] + ", it's a '"
    + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")
"""


# FINDING THE VARIOUS DIMENSIONS
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of test examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))


# RESHAPE THE TRAINING AND TEST EXAMPLES
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

print ("train_set_x_flatten shape: " + str(train_set_x_flatten.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x_flatten shape: " + str(test_shape_x_flatten.shape))
print ("test_set_y shape: " + str(test_set_y.shape))
print ("Sanity check after reshaping: " + str(train_set_x_flatten[0:5, 0]))


# STANDARDIZING THE DATASET
train_set_x = train_set_x_flatten/255
test_set_x = test_set_x_flatten/255


# SIGMOID FUNCTION
def sigmoid(z):
    """
    Compute the sigmoid of z

    Agruments:
    z -- A scalar or a numpy array of any size

    Returns:
    s -- sigmoid(z)
    """

    s = 1/(1+np.exp(-z))

    return s


"""
# function check
print ("sigmoid([0, 2]) = " + str(sigmoid(np.array([0, 2]))))
# expected output -- sigmoid([0, 2]) = [ 0.5 0.88079708]
"""


# INITIALIZE THE PARAMETERS
def initialize_with_zeros(dim):
    """
    This function creates a vector of zeros of shape (dim, 1) for w and
    initializes b to 0

    Arguments:
    dim -- size of the w vector we want (or the number of parameters in this
    case)

    Returns:
    w -- initialized vector of shape (dim, 1)
    b -- initialized scalar (corresponds to the bias)
    """

    w = np.zeros((dim, 1))
    b = 0

    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))

    return w, b


"""
# function check
dim = 2
w, b = initialize_with_zeros(dim)
print ("w = " + str(w))
print ("b = " + str(b))
# expected output -- w = [0 0]
#                    b = 0
"""


# PROPAGATE
def propagate(w, b, X, Y):
    """
    Implements the cost function and its gradient for the propogation.

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat) of
    size (1, number of examples)

    Reteurns:
    cost -- negative log likelihood cost for logistic regression
    dw -- gradient of the cost with respect to w, thus same shape as w
    db -- gradient of the cost with respect ot b, thus same shape as b
    """

    m = X.shape[1]

    # FORWARD PROPAGATION (FROM X TO COST)
    A = sigmoid(np.dot(w.T, X) + b)  # compute Activation
    cost = (-1/m)*(np.sum((Y*np.log(A) + (1 - Y)*np.log(1 - A)), axis=1))

    # BACK PROPAGATION (TO FIND GRADIENT)
    dw = (1/m) * np.dot(X, (A-Y).T)
    db = (1/m) * np.sum((A-Y), axis=1)

    assert(dw.shape == w.shape)
    assert(db.shape == b.shape)
    cost = np.squeeze(cost)
    assert(cost.shape == ())

    grads = {"dw": dw,
             "db": db}

    return grads, cost


"""
# function check
w, b, X, Y = np.array([[1.], [2.]]), 2., np.array(
    [[1., 2., -1.], [3., 4., -3.2]]), np.array([[1, 0, 1]])
grads, cost = propagate(w, b, X, Y)
print ("dw = " + str(grads["dw"]))
print ("db = " + str(grads["db"]))
print ("cost = " + str(cost))
# expected output -- dw 	[[ 0.99845601] [ 2.39507239]]
#                    db 	0.00145557813678
#                    cost 	5.801545319394553
"""


# PREDICT
def predict(w, b, X):
    """
    Predict whether the label is 0 or 1 using the learned logistic regression
    parameters (w, b)

    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of size (num_px * num_px * 3, number of examples)

    Returns:
    Y_prediction -- a numpy array (vector) containing all the predictions (0/1)
    for the examples in X
    """

    m = X.shape[1]
    Y_prediction = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)

    # COMPUTE VECTOR "A" PREDICTING THE PROBABILITIES OF THE CAT BEING IN THE
    # PICTURE

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(A.shape[1]):
        # convert probabilities A[0, i] to actual predictions p[0, i]
        Y_prediction = np.where(A > .5, 1, 0)

    assert(Y_prediction.shape == (1, m))

    return Y_prediction


# MODEL
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000,
          learning_rate=0.5, print_cost=False):
    """
    Builds the logistic regression model.

    Arguments:

    X_train -- training set represented by a numpy array of shape
    (num_px * num_px * 3, m_train)

    Y_train -- training labels represented by a numpy array (vector) of
    shape (1, m_train)

    X_test -- test set represented by a numpy array of shape
    (num_px * num_px * 3, m_test)

    Y_test --test labels represented by a numpy array (vector) of shape
    (1, m_test)

    num_iterations -- hyperparameter representing the number of
    iterations to optimize the parameters

    learning_rate -- hyperparameter representing the learning rate used
    in the update rule of optimize()

    print_cost -- set to True to print the cost every 100 iterations


    Returns:
    d -- dictionary containing the information about the model
    """

    # initialize parameters with zeros
    w, b = initialize_with_zeros(X_train.shape[0])

    # gradient descent
    parameters, grads, cost = opotimize(w, b, X_train, Y_train, num_iterations,
                                        learning_rate, print_cost=False)

    # retrive parameters w and b from dictionary "parameters"
    w = parameters["w"]
    b = parameters["b"]

    # predict train/test set examples
    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    # print train/test errors
    print ("train accuracy: {} %".format(100 - np.mean(np.abs(
        Y_prediction_train - Y_train)) * 100))
    print ("test accuracy : {} %".format(100 - np.mean(np.abs(
        Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations
         }

    return d
