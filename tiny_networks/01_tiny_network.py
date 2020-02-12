import numpy as np
np.random.seed(1)

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))


def error_deriv(y, y_hat):
    return y_hat - y


# Inputs - each row is a single training example
X = np.array([
    [0,0,1],
    [0,1,1],
    [1,0,1],
    [1,1,1]
])

# Correct outputs/labels, each row is a label 
y = np.array([
    [0],
    [0],
    [1],
    [1]
])

# random weights (3 rows, 1 cols), representing a single neuron in this layer
syn0 = 2*np.random.random((3,1)) - 1
# [[-0.16595599]
#  [ 0.44064899]
#  [-0.99977125]]
# this is the first layer of weights, connecting l0 to l1
# therefore the columns are the weights going into a single neuron. When we dot l0 and syn0, we get the activations l1, in the same format as the input (i.e. each row is the activations from a single input datum)
print('starting weights\n', syn0)

for i in range(100):
    print()
    l0 = X
    print('lo\n', l0)
    # prediction step
    l1_net = np.dot(l0, syn0)
    print('l1_net\n', l1_net)
    l1_out = nonlin(l1_net)
    print('l1_out\n', l1_out)
    print('predictions:\n', l1_out)
    # error in our prediction (compare label to prediction)
    l1_error = 0.5*(y - l1_out)**2

    print('Iteration', i, 'mean squared errors\n', l1_error)

    # l1_delta is the error weighted derivatives
    # for each sample
    l1_delta = error_deriv(y, l1_out) * nonlin(l1_out, True)
    print('l1 delta\n', l1_delta)

    print('the weight update\n', np.dot(l0.T, l1_delta))
    syn0 -= np.dot(l0.T, l1_delta)
    print('new weights\n', syn0)

print('\n\n\nl1_out\n', l1_out)
print('\n\n\nending weights\n', syn0)