import collections
import numpy
import random
import sys


def partition_dataset(Y, cvFolds, isStratified=False):
    """Partition a dataset into CV folds.

    :param Y:
    :type Y:
    :param cvFolds:
    :type cvFolds:
    :param isStratified:
    :type isStratified:
    :return :
    :rtype :

    """

    # Determine dimensions of data.
    yDimensions = Y.shape
    if len(yDimensions) == 1:
        # There is only one response variable (PLS1).
        numObservations = Y.shape[0]
        numResponses = 1
    else:
        # There are multiple response variables (PLS2).
        [numObservations, numResponses] = yDimensions

    if isStratified:
        # Create stratified partitions.        
        indicesOfClasses = []  # Class of each observation. indicesOfClasses[i] indicates the class of the ith observation.
        if numResponses == 1:
            # Y is a column vector, so determine classes from the unique values of Y.
            differentValues = numpy.unique(Y)
            if differentValues.shape[0] == 1:
                print("Stratified CV was requested, but only one class was found.")
                sys.exit()

            # Determine the number of the class each observation belongs to.
            classMembership = Y.tolist()  # Create the list to hold the class number for each observation.
            classMapping = dict([(i, ind) for ind, i in enumerate(differentValues)])  # Mapping from values in Y to class number.
            classMembership = [classMapping[i] for i in classMembership]
        else:
            # PLS2 is being performed. Each response variable is taken to be a class.
            nonzeroResponse = numpy.nonzero(Y)
            rowsWithValues = nonzeroResponse[0]
            if len(set(rowsWithValues)) != len(rowsWithValues):
                # There is at least one row with a value for multiple response variables. Stratified CV can therefore not be performed.
                print("Stratified CV was requested, but there are observations with values for multiple response variables. There class can not be determined.")
                sys.exit()

            # Determine the number of the class each observation belongs to.
            classMembership = nonzeroResponse[1].tolist()

        # Generate a warning if any class has too few observations to be in all folds.
        for i in set(classMembership):
            occurences = classMembership.count(i)
            if occurences < cvFolds:
                print("WARNING: class {0:d} occurs {1:d} times, and will not appear in each of the {2:d} folds.".format(i, occurences, cvFolds))

        # Determine the partitions.
        # Start with a list of class memberships -> [0, 1, 2, 0, 0, 1, 0, 2, 1, 0, 2, 0, 0, 1, 2, 1, 0, 0]
        # Create dictionary mapping class to indices -> {0 : [0, 3, 4, 6, 9, 11, 12, 16, 17], 1 : [1, 5, 8, 13, 15], 2 : [2, 7, 10, 14]}
        # Randomise each classes indices -> {0 : [4, 6, 9, 3, 17, 16, 0, 12, 11], 1 : [13, 1, 5, 15, 8], 2 : [10, 14, 7, 2]}
        # Split each class into cvFolds partitions -> {0 : [[0, 6, 12], [3, 9, 16], [4, 11, 17]], 1 : [[1, 13], [5, 15], [8]], 2 : [[2, 14], [7], [10]]}
        #     The overall partitions can be determined from this. For example, the first partition will be [0, 6, 12, 1, 13, 2, 4].
        # Assign partition groupings according to original indices -> [0, 0, 0, 1, 2, 1, 0, 1, 2, 1, 2, 2, 0, 0, 0, 1, 1, 2]
        #
        # Folds are only guaranteed to have a bound on the difference in the number of observations they contain.
        # The largest possible difference in size between the fold with the most observations and the one with the least is equal to the number of classes.
        #     Classes -> {0 : [[1, 2], [3]], 1 : [[4, 5], [6]], 2 : [[7, 8], [9]]}
        #     Partition -> [0, 0, 1, 0, 0, 1, 0, 0, 1]
        classIndices = collections.defaultdict(list)  # Indices of the observations belonging to each class.
        for ind, i in enumerate(classMembership):
            classIndices[i].append(ind)
        partition = [0] * numObservations  # Create the list to hold the partition number for each observation.
        for i in classIndices:
            random.shuffle(classIndices[i])  # Randomise the list of observations belonging to each class.
            classIndices[i] = [classIndices[i][j::cvFolds] for j in range(cvFolds)]  # Partition each class into cvFolds different groups.
            for ind, j in enumerate(classIndices[i]):
                for k in j:
                    partition[k] = ind
    else:
        # Create random partitions where each partition has an equal number of observations.
        # Start with a list of the indices of the observations ->  [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        # Randomise the list -> [7, 4, 8, 1, 5, 6, 9, 2, 3, 0]
        # Partition the indices -> [[7, 1, 9, 0], [4, 5, 2], [8, 6, 3]] (if cvFolds == 3)
        # Assign partition groupings according to original indices -> [0, 0, 1, 2, 1, 1, 2, 0, 2, 0]
        observationIndices = list(range(numObservations))  # List containing the index of each observation.
        random.shuffle(observationIndices)  # Randomise the order of the indices.
        partitionedIndices = [observationIndices[i::cvFolds] for i in range(cvFolds)]  # Populate each partition with every nth observation.
        partition = [0] * numObservations  # Create the list to hold the partition number for each observation.
        for ind, i in enumerate(partitionedIndices):
            for j in i:
                partition[j] = ind

    return partition