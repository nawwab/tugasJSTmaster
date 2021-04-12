import numpy as np
from activation_functions import threshold_bipolar

test1 = [
    {
        'inputs': [
            -1, -1, 1, 1, -1, -1, -1,
            -1, -1, -1, 1, -1, -1, -1,
            -1, -1, -1, 1, -1, -1, -1,
            -1, -1, 1, -1, 1, -1, -1,
            -1, -1, 1, -1, 1, -1, -1,
            -1, 1, 1, 1, 1, 1, -1,
            -1, 1, -1, -1, -1, 1, -1,
            -1, 1, -1, -1, -1, 1, -1,
            1, 1, 1, -1, 1, 1, 1
        ],
        'target': 1
    },
    {
        'inputs': [
            1, 1, 1, 1, 1, 1, -1,
            -1, 1, -1, -1, -1, -1, 1,
            -1, 1, -1, -1, -1, -1, 1,
            -1, 1, -1, -1, -1, -1, 1,
            -1, 1, 1, 1, 1, 1, -1,
            -1, 1, -1, -1, -1, -1, 1,
            -1, 1, -1, -1, -1, -1, 1,
            -1, 1, -1, -1, -1, -1, 1,
            1, 1, 1, 1, 1, 1, -1
        ],
        'target': -1
    },
    {
        'inputs': [
            -1, -1, 1, 1, 1, 1, 1,
            -1, 1, -1, -1, -1, -1, 1,
            1, -1, -1, -1, -1, -1, -1,
            1, -1, -1, -1, -1, -1, -1,
            1, -1, -1, -1, -1, -1, -1,
            1, -1, -1, -1, -1, -1, -1,
            1, -1, -1, -1, -1, -1, -1,
            -1, 1, -1, -1, -1, -1, 1,
            -1, -1, 1, 1, 1, 1, -1
        ],
        'target': -1
    } 
]

test2 = [
    {
        'inputs': [
            -1, -1, -1, 1, -1, -1, -1,
            -1, -1, -1, 1, -1, -1, -1,
            -1, -1, -1, 1, -1, -1, -1,
            -1, -1, 1, -1, 1, -1, -1,
            -1, -1, 1, -1, 1, -1, -1,
            -1, 1, -1, -1, -1, 1, -1,
            -1, 1, 1, 1, 1, 1, -1,
            -1, 1, -1, -1, -1, 1, -1,
            -1, 1, -1, -1, -1, 1, -1,
        ],
        'target': 1
    },
    {
        'inputs': [
            1, 1, 1, 1, 1, 1, -1,
            1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, -1, -1, -1, 1,
            1, 1, 1, 1, 1, 1, -1,
            1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, -1, -1, -1, 1,
            1, 1, 1, 1, 1, 1, -1,
        ],
        'target': -1
    },
    {
        'inputs': [
            -1, -1, 1, 1, 1, -1, -1,
            -1, 1, -1, -1, -1, 1, -1,
            1, -1, -1, -1, -1, -1, 1,
            1, -1, -1, -1, -1, -1, -1,
            1, -1, -1, -1, -1, -1, -1,
            1, -1, -1, -1, -1, -1, -1,
            1, -1, -1, -1, -1, -1, 1,
            -1, 1, -1, -1, -1, 1, -1,
            -1, -1, 1, 1, 1, -1, -1,
        ],
        'target': -1
    }
]

def notA(test_data, learn_rate, activation_func, threshold=0):
    weights = np.zeros_like(test_data[0]['inputs'])
    inputs = np.array(test_data[0]['inputs'])
    bias = 0
    epoch = 0

    while True:
        ready = True
        epoch += 1
        print(f"epoch: {epoch}")
        for data in test_data:
            inputs = np.array(data['inputs'])
            f_net = activation_func(np.dot(weights, inputs) + bias, threshold)
            if f_net != data['target']:
                ready = False
                delta_weights = np.array([*(inputs * learn_rate * data['target'])])
                delta_bias = learn_rate * data['target']
                weights = weights + delta_weights
                bias = bias + delta_bias
            else:
                continue

        if ready:
            break

    return {
        'weights': weights,
        'bias': bias,
        'epoch': epoch
    }

print(notA(test1 + test2, 0.1, threshold_bipolar, 0.5)['weights'])