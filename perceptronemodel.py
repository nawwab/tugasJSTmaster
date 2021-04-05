from basicNN import NN
from inputs import bipolar_tuples, biner_tuples
from activation_functions import threshold_bipolar
from tests_data import bipolar_AND_test_data, bipolar_OR_test_data, bipolar_biner_AND_test_data

# perceptron model, returning new weights and bias
def perceptron_model(test_data, learn_rate, activation_func, threshold=0):
    weight1 = 0
    weight2 = 0
    bias = 0
    epoch = 0
    while True:
        ready = True
        epoch += 1
        print(f"epoch: {epoch}")
        for data in test_data: 
            input1 = data['input1']
            input2 = data['input2']
            f_net = NN(input1, weight1, input2, weight2, activation_func, bias, threshold)
            if f_net != data['target']:
                ready = False
                delta_w1 = learn_rate * data['target'] * input1
                delta_w2 = learn_rate * data['target'] * input2
                delta_bias = learn_rate * data['target']
                weight1 += delta_w1
                weight2 += delta_w2
                bias += delta_bias
            else:
                continue
        
        if ready:
            break
    
    return {
        'weight1': weight1,
        'weight2': weight2,
        'bias': bias,
        'epoch': epoch
    }


print("AND, learn rate = 1, threshold = 0")
# get the weights and bias
weight1, weight2, bias, epoch = perceptron_model(bipolar_AND_test_data, 1, threshold_bipolar, 0).values()
# test the hebb model with bipolar inputs
print(f"|{'input1'.center(10, ' ')}|{'input2'.center(10, ' ')}|{'result'.center(10, ' ')}|")
for some_inputs in bipolar_tuples:
    input1, input2 = some_inputs
    result = NN(input1, weight1, input2, weight2, threshold_bipolar, bias)
    print(f"|{str(input1).center(10, ' ')}|{str(input2).center(10, ' ')}|{str(result).center(10, ' ')}|")

print("OR, learn rate = 1, threshold = 0")
# get the weights and bias
weight1, weight2, bias, epoch = perceptron_model(bipolar_OR_test_data, 1, threshold_bipolar, 0).values()
# test the hebb model with bipolar inputs
print(f"|{'input1'.center(10, ' ')}|{'input2'.center(10, ' ')}|{'result'.center(10, ' ')}|")
for some_inputs in bipolar_tuples:
    input1, input2 = some_inputs
    result = NN(input1, weight1, input2, weight2, threshold_bipolar, bias)
    print(f"|{str(input1).center(10, ' ')}|{str(input2).center(10, ' ')}|{str(result).center(10, ' ')}|")

print("AND - biner bipolar, learn rate=0.2")
# get the weights and bias
weight1, weight2, bias, epoch = perceptron_model(bipolar_biner_AND_test_data, 1, threshold_bipolar, 0.2).values()
# test the hebb model with bipolar inputs
print(f"|{'input1'.center(10, ' ')}|{'input2'.center(10, ' ')}|{'result'.center(10, ' ')}|")
for some_inputs in biner_tuples:
    input1, input2 = some_inputs
    result = NN(input1, weight1, input2, weight2, threshold_bipolar, bias, 0.2)
    print(f"|{str(input1).center(10, ' ')}|{str(input2).center(10, ' ')}|{str(result).center(10, ' ')}|")
