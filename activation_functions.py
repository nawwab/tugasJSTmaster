def threshold_bipolar(input, threshold=0):
    return 1 if input >= threshold else -1

def threshold_biner(input, threshold=0):
    return 1 if input >= threshold else 0
