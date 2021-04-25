# Model neuron McCulloch-Pitts
from activation_functions import threshold_biner
from inputs import biner_tuples
from basicNN import NN

# kasus x1 OR x2
print("x1 OR x2")
for x1, x2 in biner_tuples:
    result = NN(x1, 1, x2, 1, threshold_biner, 0, 1)
    print(f"{x1} OR {x2}: {result}")

print()

# kasus x1 AND x2
for x1, x2 in biner_tuples:
    result = NN(x1, 1, x2, 1, threshold_biner, 0, 2)
    print(f"{x1} AND {x2}: {result}")

print()

# kasus x1 AND Not x2
for x1, x2 in biner_tuples:
    result = NN(x1, 2, x2, -1, threshold_biner, 0, 2)
    print(f"{x1} AND not {x2}: {result}")

print()

# kasus Not x1 AND x2
for x1, x2 in biner_tuples:
    result = NN(x1, -1, x2, 2, threshold_biner, 0, 2)
    print(f"not {x1} AND {x2}: {result}")

print()

# kasus x1 XOR x2 
for x1, x2 in biner_tuples:
    z1 = NN(x1, 2, x2, -1, threshold_biner, 0, 2)
    z2 = NN(x1, -1, x2, 2, threshold_biner, 0, 2)
    result = NN(z1, 1, z2, 1, threshold_biner, 0, 1)
    print(f"{x1} XOR {x2}: {result}")

print()
