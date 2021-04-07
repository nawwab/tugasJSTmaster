from basicNN import custom_NN
from inputs import bipolar_tuples
from activation_functions import threshold_bipolar
from tests_data import bipolar_AND_test_data

def identity_func(x):
    y = x
    return y

# adaline model, returning new weights and bias
def perceptron_model(test_data, learn_rate, tolerance_value):
    weights = [0 for i in range(len(test_data[0].values()) - 1)]
    bias = 0
    epoch = 0
    iter_flag = True

    while iter_flag:
        epoch += 1
        print(f"epoch: {epoch}")
        old_weights = [*weights]

        for entry in test_data:
            data = list(entry.values())
            inputs = data[:-1]
            target = data[-1]
            f_net = custom_NN(inputs, weights, identity_func, bias)

            if f_net != target:
                ready = False
                weights = [
                    weight_val + (learn_rate * (target - f_net) * input_val) for input_val, weight_val in zip(inputs, weights)
                ]
                bias += learn_rate * (target - f_net)
            else:
                continue
        
        for old_weight, weight in zip(old_weights, weights):
            if (old_weight - weight) ** 2 > tolerance_value:
                break
            else:
                iter_flag = False

        if not iter_flag:
            break
    
    return [ weights, bias, epoch ]

print("AND, learn rate = 0.1, threshold = 0, tolerance_val = 0.05")
# get the weights and bias
weights, bias, epoch = perceptron_model(bipolar_AND_test_data, 0.1, 0.05)
# test the hebb model with bipolar inputs
for i in range(len(list(bipolar_tuples[0]))):
    print(f"| {f'input{i}'.center(10, ' ')}", end="")
print(f"|{'result'.center(10, ' ')}|")

for entry in bipolar_tuples:
    inputs = list(entry)
    result = custom_NN(inputs, weights, threshold_bipolar, bias, 0)
    for input_val in inputs:
        print(f"| {str(input_val).center(10, ' ')}", end="")
    print(f"|{str(result).center(10, ' ')}|")
