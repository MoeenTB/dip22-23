def num_gen(dims : tuple, seed : int):
    '''
    dims holds the dimensions of the matrix
    '''
    if (not isinstance(seed, int)):
        raise TypeError("The seed (seed) must be entered as an int.")
    if (not isinstance(dims, tuple) or len(dims)!= 2):
        raise TypeError("Dimensions (dims) must be entered as a tuple of length 2.")

    output = []
    temp = [0] * (dims[1]+1)
    for i in range(dims[0]+1):
        output.append(temp.copy())

    output[1][1] = seed
    for i in range(1, dims[0]+1):
        for j in range(1, dims[1]+1):
            if (i == 1) and (j == 1):
                continue
            output[i][j] = output[i][j-1] - output[i-1][j] - output[i-1][j-1]
            
    output = output[1:]
    for i in range(dims[0]):
        del(output[i][0])
    return output

dims = (3, 4)
output = num_gen(dims, 1)
print(output)
