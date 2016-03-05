import numpy
import PLS.center_and_store
from scipy import sparse
import unittest


class CompletionTests(unittest.TestCase):
    """Tests checking whether the matrix storage code successfully returns."""

    @classmethod
    def setUpClass(cls):
        """Setup inputs needed for tests in the class."""

        cls.smallRowMat = cls.xSmall = numpy.array([[4, 2, 10], [1, 5, 12], [7, 11, 9], [3, 6, 8]])
        cls.smallColMat = cls.xSmall = numpy.array([[4, 2, 17, 10, 13], [16, 1, 5, 18, 12], [7, 11, 19, 9, 14], [15, 20, 3, 6, 8]])

    def test_pass_small(self):
        PLS.center_and_store.center_and_store(self.smallRowMat, "TestMoreRowsLoc.txt")
        PLS.center_and_store.center_and_store(self.smallColMat, "TestMoreColsLoc.txt")


class CorrectnessTests(unittest.TestCase):
    """Tests checking the correctness of the output."""

    pass


if __name__ == '__main__':
    unittest.main()