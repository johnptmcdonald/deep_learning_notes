import numpy as np

X = np.array([
    [0, 0, 1, 1],
    # [0, 1, 1, 1],
    # [1, 0, 1, 0],
    # [1, 1, 1, 0],
    # [1, 0, 1, 0],
    # [1, 1, 1, 0]
])

y = np.array([
    [0],
    # [0],
    # [1],
    # [1],
    # [1],
    # [1],    
])

def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    else:
        return 1/(1+np.exp(-x))

W01 = np.random.random((4,3))
W12 = np.random.random((3,2))
W23 = np.random.random((2,1))

lr = 0.05

for i in range(1000):
    print()
    # feedforward
    l0 = X
    l1 = nonlin(X.dot(W01))
    l2 = nonlin(l1.dot(W12))
    l3 = nonlin(l2.dot(W23))

    error = (0.5 * (y - l3)**2)/4
    print('ERROR after iteration ', i, ': ', error)

    # backprop - must transpose weights
    delta_l3 = (l3 - y) * l3*(1 - l3)
    delta_l2 = delta_l3.dot(W23.T) * l2*(1-l2)    
    delta_l1 = delta_l2.dot(W12.T) * l1*(1-l1)

    # Each neuron has a delta that represents how changing the raw activation of that neuron changes the error (for all data points). So if the input data has 100 rows, the delta for a neuron will have 100 rows. For delta_l2 and delta_l1 you can think of it as how much of the next layer's delta comes from this neuron. 

    # The reason you have to transpose the weights is because you want to multiply each weight by the next neuron's delta, to get this layer's neurons' deltas. 
    # e.g. for  (A)--w1-\
    #                    \
    #                    (C)
    #                    /
    #           (B)-- w2/

    # the weight matrix here is: 
    # [
    #   [w1],
    #   [w2]
    # ]

    # and the deltas for C would be a long single column vector:
    # [
    #     [delta_for_C],
    #     [delta_for_C],
    #     [delta_for_C],
    #     [delta_for_C],
    #     [delta_for_C],
    #     ...
    # ]

    # and we want to get the deltas for A and B like this:
    # [
    #     [delta_for_A, delta_for_B],
    #     [delta_for_A, delta_for_B],
    #     [delta_for_A, delta_for_B],
    #     [delta_for_A, delta_for_B],
    #     [delta_for_A, delta_for_B],
    #     ...
    # ]

    # which is equal to...
    # [
    #     [w1 * delta_for_C, w2 * delta_for_C],
    #     [w1 * delta_for_C, w2 * delta_for_C],
    #     [w1 * delta_for_C, w2 * delta_for_C],
    #     [w1 * delta_for_C, w2 * delta_for_C],
    #     [w1 * delta_for_C, w2 * delta_for_C],
    #     ...
    # ]


    # update weights must transpose layer outputs
    print('l0', l0.shape, l0)
    print('l0.T', l0.T.shape, l0.T)
    print('delta_l1', delta_l1.shape, delta_l1)
    print('l0.T.dot(delta_l1)', l0.T.dot(delta_l1))
    print('W01', W01)
    W23 -= l2.T.dot(delta_l3)
    W12 -= l1.T.dot(delta_l2)
    W01 -= l0.T.dot(delta_l1)

    # Again, we need to transpose because of the format of the inputs/activations and the deltas. For deltas, each column is the deltas for a single neuron in a layer. For activations, each row is an activation vector of the previous layer. 
    
    # e.g. for the first layer and a single data point:

    # input activation = [[0, 0, 1, 1]]
    # transposed input_activation = [
    #     [0],
    #     [0],
    #     [1],
    #     [1]
    # ]
    # layer 1 deltas (3 neurons) = [[-0.1, 0.3, 0.25]]
    # we want to change the weights, which currently look like this:
    
    # W01 = [
    #         [1, 2, 3],
    #         [3, 2, 4],
    #         [2, 1, 3],
    #         [2, 3, 1]
    # ]        ^  ^  ^
    #          |  |  | 
    #          |  |  weights going into neuron 3
    #          |  |
    #          | weights going into neuron 2
    #          |
    #         weights going into neuron 1 

    # so each activation into a neuron needs to be multiplied by the delta of that neuron. That product gives you the change in the weight for that 'path' of activation into that neuron. We sum all the votes to change a path's weights.

l0 = X
l1 = nonlin(np.dot(l0,W01))
l2 = nonlin(np.dot(l1,W12))
l3 = nonlin(np.dot(l2,W23))

print(l3)

    

