from basicNN import NN
from inputs import bipolar_tuples
from activation_functions import threshold_bipolar
from tests_data import bipolar_AND_test_data

# hebb model, returning new weights and bias
def hebb_model(input_dicts):
    input1 = 0
    input2 = 0
    bias = 0
    for data in input_dicts:
        input1 += data['input1'] * data['target']
        input2 += data['input2'] * data['target']
        bias += 1 * data['target']
    return {
        'input1': input1,
        'input2': input2,
        'bias': bias
    }

# get the weights and bias
weight1, weight2, bias = hebb_model(bipolar_AND_test_data).values()
# test the hebb model with bipolar inputs
print(f"|{'input1'.center(10, ' ')}|{'input2'.center(10, ' ')}|{'result'.center(10, ' ')}|")
for some_inputs in bipolar_tuples:
    input1, input2 = some_inputs
    # set a neural network that using threshold_bipolar as activation function
    # with 2 as a threshold value
    result = NN(input1, weight1, input2, weight2, threshold_bipolar, bias, 2)
    print(f"|{str(input1).center(10, ' ')}|{str(input2).center(10, ' ')}|{str(result).center(10, ' ')}|")
