def center_and_store(matrix, fileMatrix, fileMatrixTranspose):
    """Center a matrix and save the result.

    Normally centering a matrix will cause a sparse matrix to become dense.
    This method will center and store a matrix (thereby causing it to become dense) without ever keeping the
    entire matrix in memory.

    :param matrix:                  The matrix to be centered and stored
    :type matrix:                   numpy/scipy 2D array or matrix (or similar type exposing shape, mean and with indexing)
    :param fileMatrix:              The location to save the centered matrix
    :type fileMatrix:               string
    :param fileMatrixTranspose:     The location to save the transpose of the centered matrix
    :type fileMatrixTranspose:      string

    """

    # Calculate the mean of the matri's columns.
    matrixMean = matrix.mean(axis=0)

    # Determine whether to save the centered matrix so that one row is on each line of the file or one so that
    # one column is on each line of the file. This will be done based on which dimension is smallest.
    [numRows, numCols] = matrix.shape
    with open(fileMatrix, 'w') as writeMatrix:
        # Save the matrix with a row on each line.
        for i in range(numRows):
            centeredRow = matrix[i, :] - matrixMean  # Center the row.
            centeredRow.tofile(writeMatrix, sep='\t')
            writeMatrix.write('\n')
    with open(fileMatrixTranspose, 'w') as writeMatrixTranspose:
        # Save the matrix with a column on each line.
        for i in range(numCols):
            centeredCol = matrix[:, i] - matrixMean[i]  # Center the column.
            centeredCol.reshape(numRows, 1)  # Reshape it to a column array (so it can be transposed).
            centeredCol = centeredCol.T
            centeredCol.tofile(writeMatrixTranspose, sep='\t')
            writeMatrixTranspose.write('\n')