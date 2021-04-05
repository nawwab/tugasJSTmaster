# test data for AND operator
bipolar_AND_test_data = [
    {
        'input1': -1,
        'input2': -1,
        'target': -1
    },
    {
        'input1': -1,
        'input2': 1,
        'target': -1
    },
    {
        'input1': 1,
        'input2': -1,
        'target': -1
    },
    {
        'input1': 1,
        'input2': 1,
        'target': 1
    }
]

biner_AND_test_data = [
    {
        'input1': 0,
        'input2': 0,
        'target': 0
    },
    {
        'input1': 0,
        'input2': 1,
        'target': 0
    },
    {
        'input1': 1,
        'input2': 0,
        'target': 0
    },
    {
        'input1': 1,
        'input2': 1,
        'target': 1
    }
]

# test data for OR operator
bipolar_OR_test_data = [
    {
        'input1': -1,
        'input2': -1,
        'target': -1
    },
    {
        'input1': -1,
        'input2': 1,
        'target': 1
    },
    {
        'input1': 1,
        'input2': -1,
        'target': 1
    },
    {
        'input1': 1,
        'input2': 1,
        'target': 1
    }
]

biner_OR_test_data = [
    {
        'input1': 0,
        'input2': 0,
        'target': 0
    },
    {
        'input1': 0,
        'input2': 1,
        'target': 1
    },
    {
        'input1': 1,
        'input2': 0,
        'target': 1
    },
    {
        'input1': 1,
        'input2': 1,
        'target': 1
    }
]

bipolar_biner_AND_test_data = [
    {
        'input1': 0,
        'input2': 0,
        'target': -1
    },
    {
        'input1': 0,
        'input2': 1,
        'target': -1
    },
    {
        'input1': 1,
        'input2': 0,
        'target': -1
    },
    {
        'input1': 1,
        'input2': 1,
        'target': 1
    }
]
