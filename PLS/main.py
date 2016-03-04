import argparse
import numpy
import PLS.simpls
import random

def pls(X, Y, numberComponents=10, cvFolds=0, cvMethod="MSE", isCVStratified=True, isMemUsed=True):
    """Perform PLS using the SIMPLS algorithm.

    Stuff about if X is sparse then it will try to make the matrix dense unless isMemUsed is false.

    :param X:
    :type X:
    :param Y:
    :type Y:
    :param numberComponents:
    :type numberComponents:
    :param cvFolds:
    :type :cvFolds:
    :param cvMethod:
    :type cvMethod:
    :param isCVStratified:
    :type isCVStratified:
    :param isMemUsed:
    :type isMemUsed:
    :returns :
    :type :

    If CV is used, then it will return the partitioning
    Stratified partitioning only makes sense in cases of classification
        For PLS1, observations will be grouped by their response value (the number of groups will be the number of unique response values)
        For PLS2, each response variable will be treated as a class. Observations will be grouped based on nonzero values in the response variable.
            If an observation has a nonzeo value for more than one response variable and error will be thrown.

    """

    # Determine dimensions of inputs.
    [numObservationsX, numPredictors] = X.shape
    yDimensions = Y.shape
    if len(yDimensions) == 1:
        # There is only one response variable (PLS1).
        numObservationsY = Y.shape[0]
        numResponses = 1
    else:
        # There are multiple response variables (PLS2).
        [numObservationsY, numResponses] = Y.shape

    #========================================#
    # Process and validate the user's input. #
    #========================================#
    errorsFound = []  # List recording all error messages to display.

    if numObservationsX != numObservationsY:
        # X and Y must have the same number of rows.
        errorsFound.append("The first dimension of X and Y are not equal ({0:d} and {1:d}).".format(numObservationsX, numObservationsY))

    if numberComponents < 1:
        # There must be at least one hidden component used.
        errorsFound.append("The number of components must be at least one.")
    maxNumComponents = min(numObservationsX - 1, numPredictors)
    if numberComponents > maxNumComponents:
        # You can't have more components than the smaller of the two dimensions of X.
        errorsFound.append("The maximum number of components is {0:d}.".format(maxNumComponents))

    isCVUsed = cvFolds != 0
    if isCVUsed and (cvFolds < 2):
        # There must be at least 2 CV folds.
        errorsFound.append("If a non-zero value is provided for the number of CV folds, then the number must be at least two.")
    # TODO add checking that the cvMethod is one of "MSE", "EqualError" or a user supplied function meeting some to be decided criteria

    ###############################
    # Run appropriate PLS method. #
    ###############################
    # Center the data.
    meanX = X.mean(axis=0)
    meanY = Y.mean(axis=0)
    if isMemUsed:
        # Center the data in memory.
        X = X - meanX
        Y = Y - meanY
    else:
        # TODO add centering via the not in memory method
        pass

    if isCVUsed:
        # Run PLS using cross validation.
        # TODO put in the cross validation running and determination of folds
        if isMemUsed:
            # Run CV using the memory-based SIMPLS.
            returnObject = {}
        else:
            # Run CV using the file system-based SIMPLS.
            returnObject = {}
    else:
        # Run PLS without cross validation.
        if isMemUsed:
            # Run SIMPLS without resorting to the file system.
            xLoadings, yLoadings, xScores, yScores, weights = PLS.simpls.simpls(X, Y, numberComponents)

            # Calculate coefficients.
            coefficients = weights.dot(yLoadings.T)
            intercept = meanY - (meanX.dot(coefficients))
            coefficients = numpy.vstack((intercept, coefficients))

            # Calculate the percentage of the variance of X and Y that is explained.
            xPercentVarExp = sum(numpy.square(abs(xLoadings))) / sum(sum(numpy.square(abs(X))))
            yPercentVarExp = sum(numpy.square(abs(yLoadings))) / sum(sum(numpy.square(abs(Y))))

            # Setup the object used to return the results.
            returnObject = {}
            returnObject["xLoadings"] = xLoadings
            returnObject["yLoadings"] = yLoadings
            returnObject["xScores"] = xScores
            returnObject["yScores"] = yScores
            returnObject["weights"] = weights
            returnObject["coefficients"] = coefficients
            returnObject["xPercentVarExp"] = xPercentVarExp
            returnObject["yPercentVarExp"] = yPercentVarExp
        else:
            # Run SIMPLS using the file system.
            # TODO add the file system SIMPLS call
            returnObject = {}

    return returnObject


def pls_cv_mem(X, Y, numberComponents=10, cvFolds=0, cvMethod="MSE", isCVStratified=True):
    """Memory-based cross validation for PLS.

    Performs no error checking on inputs. If error checking is desired, use the pls function.

    :param X:
    :type X:
    :param Y:
    :type Y:
    :param numberComponents:
    :type numberComponents:
    :param cvFolds:
    :type :cvFolds:
    :param cvMethod:
    :type cvMethod:
    :param isCVStratified:
    :type isCVStratified:
    :returns :
    :type :

    Stratified partitioning only makes sense in cases of classification
        For PLS1, observations will be grouped by their response value (the number of groups will be the number of unique response values)
        For PLS2, each response variable will be treated as a class. Observations will be grouped based on nonzero values in the response variable.
            If an observation has a nonzeo value for more than one response variable and error will be thrown.

    """

    # Determine dimensions of inputs.
    [numObservationsX, numPredictors] = X.shape
    yDimensions = Y.shape
    if len(yDimensions) == 1:
        # There is only one response variable (PLS1).
        numObservationsY = Y.shape[0]
        numResponses = 1
    else:
        # There are multiple response variables (PLS2).
        [numObservationsY, numResponses] = Y.shape

    # Generate the cross validation partitions.
    if isCVStratified:
        # Create stratified partitions.
        pass
    else:
        # Create random partitions where each partition has an equal number of observations.
        # Start with a list of the indices of the observations ->  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Randomise the list -> [7, 4, 8, 1, 5, 6, 9, 2, 3, 0]
        # Partition the indices -> [[7, 1, 9, 0], [4, 5, 2], [8, 6, 3]] (if cvFolds == 3)
        # Assign partition groupings according to original indices -> [0, 0, 1, 2, 1, 1, 2, 0, 2, 0]
        observationIndices = list(range(numObservationsX))  # List containing the index of each observation.
        random.shuffle(observationIndices)  # Randomise the order of the indices.
        partitionedIndices = [observationIndices[i::cvFolds] for i in range(cvFolds)]  # Populate each partition with every nth observation.
        partition = [0] * numObservationsX  # Create the list to hold the partition number for each observation.
        for ind, i in enumerate(partitionIndices):
            for j in i:
                partition[j] = ind

    # As each CV fold is smaller than the full dataset, it may not be possible to use the number of components requested.
    # Determine this and use as many as we can.