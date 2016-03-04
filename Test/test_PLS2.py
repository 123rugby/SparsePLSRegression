import numpy
import PLS.main
from scipy import sparse
import unittest

class CompletionTests(unittest.TestCase):
    """Tests checking whether the code doesn't crash while running PLS1."""

    @classmethod
    def setUpClass(self):
        """Setup inputs needed for tests in the class."""

        self.xSmall = numpy.array([[4, 2, 3], [1, 5, 8], [7, 6, 9]])
        self.ySmall = numpy.array([[1, 0], [0, 1], [1, 0]])

    def test_pass_small(self):
        PLS.main.pls(self.xSmall, self.ySmall)


class CorrectnessTests(unittest.TestCase):
    """Tests checking the correctness of the output while running PLS1."""

    pass


if __name__ == '__main__':
    unittest.main()