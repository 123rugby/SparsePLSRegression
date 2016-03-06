import numpy


def dot_product(fileX, numRowsX, Y, sep='\t'):
    """Calculate the dot product of X and Y.

    X is stored in the file fileX. If (X.T).dot(Y) needs to be calculated, then record the transpose of X in a file
    and pass that location in as the first parameter.

    There is no checking whether the dot product of the matrices is too large for memory. It is assumed that it will fit.

    :param fileX:           Location where the X matrix is stored.
    :type fileX:            string
    :param numRowsX:        The number of rows in the X matrix.
    :type numRowsX:         int
    :param Y:               The Y matrix
    :type Y:                numpy/scipy array/matrix
    :param sep:             The separator used between elements of the X matrix.
    :type sep:              string
    :return :               The dot product of the two matrices.
    :rtype :                numpy matrix

    """

    # Determine the number of columns in the Y matrix.
    yDimensions = Y.shape
    if len(yDimensions) == 1:
        # There is only one response variable (PLS1).
        numObservationsY = Y.shape[0]
        numResponses = 1
        Y = Y.reshape(numObservationsY, 1)  # Ensure that Y is a column vector.
    else:
        # There are multiple response variables (PLS2).
        [numObservationsY, numResponses] = yDimensions

    # Preallocate the result matrix.
    # The matrix resulting from a dot product between X and Y has the same number of rows as X and the
    # same number of columns as Y.
    dotProduct = numpy.matrix(numpy.empty((numRowsX, numResponses)))

    # Calculate the dot product.
    lineCount = 0
    with open(fileX, 'r') as readX:
        for line in readX:
            row = line.strip()
            row = numpy.fromstring(row, sep=sep)  # Generate a numpy array from the row.
            dotProduct[lineCount, :] = row.dot(Y)  # Calculate the dor product of the row with the Y matrix.
            lineCount += 1

    return dotProduct