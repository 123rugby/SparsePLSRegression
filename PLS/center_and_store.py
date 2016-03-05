def center_and_store(matrix, fileSaveLocation):
    """Center a matrix and save the result.

    Normally centering a matrix will cause a sparse matrix to become dense.
    This method will center and store a matrix (thereby causing it to become dense) without ever keeping the
    entire matrix in memory.
    When there are more rows than columns, the first column will be written out on the first row, the second
    on the second, etc. Each column will be written out transposed.

    :param matrix:              The matrix to be centered and stored
    :type matrix:               numpy/scipy 2D array or matrix (or similar type exposing shape, mean and with indexing)
    :param fileSaveLocation:    The location to save the centered matrix
    :type fileSaveLocation:     string
    :return :                   Whether the matrix was stored by rows or columns
    :rtype :                    boolean

    """

    # Calculate the mean of the matri's columns.
    matrixMean = matrix.mean(axis=0)

    # Determine whether to save the centered matrix so that one row is on each line of the file or one so that
    # one column is on each line of the file. This will be done based on which dimension is smallest.
    [numRows, numCols] = matrix.shape
    with open(fileSaveLocation, 'w') as writeSaveLocation:
        if numCols < numRows:
            # Save the matrix with a column on each line.
            for i in range(numCols):
                centeredCol = matrix[:, i] - matrixMean[i]  # Center the column.
                centeredCol.reshape(numRows, 1)  # Reshape it to a column array (so it can be transposed).
                centeredCol = centeredCol.T
                centeredCol.tofile(writeSaveLocation, sep='\t')
                writeSaveLocation.write('\n')
        else:
            # Save the matrix with a row on each line.
            for i in range(numRows):
                centeredRow = matrix[i, :] - matrixMean  # Center the row.
                centeredRow.tofile(writeSaveLocation, sep='\t')
                writeSaveLocation.write('\n')

    return numRows < numCols