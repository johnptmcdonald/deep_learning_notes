import numpy as np
np.random.seed(1)

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

def error_deriv(y, y_hat):
    return y_hat - y


X = np.array([
    [0,0,1],
])

# Correct outputs/labels, each row is a label 
y = np.array([
    [0],
])

# randomly initialize our weights with mean 0
# i.e. first layer is 3 neurons, second layer is 2 neuron
syn0 = np.random.random((3,4))
syn1 = np.random.random((4,1))
print('syn0', syn0)
print('syn1', syn1)

for i in range(1000):
    # 'neuron' things are columns. e.g. activations, deltas
    print()
    # Feed forward through layers 0, 1, and 2
    l0 = X
    l1_net = np.dot(l0, syn0)
    l1_out = nonlin(l1_net) 
    l2_net = np.dot(l1_out, syn1)
    l2_out = nonlin(l2_net)
    
    l2_error = np.sum(0.5 * (y - l2_out)**2)
    print('error', l2_error)

    # backprop the deltas:
    # the delta for a neuron is how changing the net activation changes the error. I.e. its contribution to total error
    delta_l2 = error_deriv(y, l2_out) * nonlin(l2_out, True)
    print('delta_l2', delta_l2)
    
    # amount of delta_l2 coming from each neuron in l1
    # keep it as a col for each neuron (i.e. a single row)
    l1_error = np.dot(delta_l2, syn1.T)

    delta_l1 = l1_error * nonlin(l1_out, True)
    # print('l0.T', l0.T)
    print('delta_l1', delta_l1)
    # print('np.dot(l0.T, delta_l1)', np.dot(l0.T, delta_l1))
    # update the weights
    print('l1\n',l1_out)
    print('l0\n', l0)
    syn1 -= np.dot(l1_out.T, delta_l2)
    syn0 -= np.dot(l0.T, delta_l1)

    # the key is that the weights are in a particular format:
        # each row in the matrix represents the weights from one neuron to another. e.g.
        #  (a)
        #         (d) 
        #  (b)             (f)
        #         (e)
        #  (c)

        # in this case W0 would be:
            # [
            #   [ad,ae],
            #   [bd,be],
            #   [cd,ce]
            # ]
        # and W1 would be:
            # [
            #   [df],
            #   [ef]
            # ]
        
        # To allow us to do weight updates in the easiest way, we should keep our deltas as rows (where each col is a delta for a neuron). Backpropping deltas means we have to transpose the weights matrix to get the backpropped delta, just because of the way the weights matrix is stored
        # Then to made any weight updates we need to make sure the update is the same shape as the weight matrix, which means we also need to transpose the activations 


l0 = X
l1 = nonlin(np.dot(l0,syn0))
l2 = nonlin(np.dot(l1,syn1))

print(l2)