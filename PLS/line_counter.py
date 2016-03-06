import itertools


def line_counter(fileLocation):
    """Count the lines in a file.

    :param fileLocation:    The location of the file.
    :type fileLocation:     string
    :return :               The count of the number of lines in the file.
    :rtype :                int

    """

    fid = open(fileLocation, 'rb')  # Open the file to read binary data.

    # Setup for reading the file.
    infiniteIterator = itertools.repeat(None)  # Create an iterator that returns None indefinitely.
    takewhilePredicate = lambda x : x  # Will ensure that takewhile stops once the line returned is "False" (i.e. once it is an EOF).

    # Create a generator that will repeat(edly) read a chunk of the file in until the line is "False" (i.e. once it is EOF).
    readGen = itertools.takewhile(takewhilePredicate, (fid.raw.read(1024*1024) for _ in infiniteIterator))

    lineCount = sum(buf.count(b'\n') for buf in readGen)  # Count occurrences of '\n' (and therefore number of lines) in each chunk of the file.
    fid.close()

    return lineCount