import numpy
import PLS.partition_dataset
import unittest


class CompletionTests(unittest.TestCase):
    """Tests checking whether the code successfully returns."""

    @classmethod
    def setUpClass(cls):
        """Setup inputs needed for tests in the class."""

        cls.ySmall = numpy.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0], [0, 0, 1], [0, 0, 1]])

    def test_pass_small(self):
        partition = PLS.partition_dataset.partition_dataset(self.ySmall, 2, False)
        partition = PLS.partition_dataset.partition_dataset(self.ySmall, 2, True)

    def test_too_many_folds(self):
        partition = PLS.partition_dataset.partition_dataset(self.ySmall, 4, False)
        partition = PLS.partition_dataset.partition_dataset(self.ySmall, 4, True)

    # Tests to add
    # CV stratified and non-stratified
    # Larger matrices
    # Y matrix with one class (PLS1)


class CorrectnessTests(unittest.TestCase):
    """Tests checking the correctness of the output."""

    pass


if __name__ == '__main__':
    unittest.main()