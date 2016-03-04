import argparse
import numpy
import simpls

def pls(X, Y, numberComponents=10, cvFolds=0, cvMethod="MSE", isCVStratified=True, isMemUsed=True):
    """Perform PLS using the SIMPLS algorithm.

    Stuff about if X is sparse and you specify isCenter then it will try to make the matrix dense unless isMemUsed is false.

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
        pass
    else:
        # Run PLS without cross validation.
        xLoadings, yLoadings, xScores, yScores, weights = simpls.simpls(X, Y, numberComponents)

        # Calculate coefficients.
        coefficients = weights.dot(yLoadings.T)
        #coefficients = 
        print(coefficients)