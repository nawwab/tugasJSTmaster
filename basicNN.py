def NN(input1, weight1, input2, weight2, activation_func, bias=0, threshold=None):
    value = (input1 * weight1) + (input2 * weight2) + bias
    if threshold is not None:
        return activation_func(value, threshold)
    else:
        return activation_func(value)

def custom_NN(inputs, weights, activation_func, bias=0, threshold=None):
    weighted = [input_val * weight_val for input_val, weight_val in zip(inputs, weights)]
    value = sum(weighted) + bias
    if threshold is not None:
        return activation_func(value, threshold)
    else:
        return activation_func(value)