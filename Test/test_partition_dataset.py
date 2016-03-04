import numpy
import PLS.partition_dataset
import numpy

class CompletionTests(unittest.TestCase):
    """Tests checking whether the code successfully returns."""

    @classmethod
    def setUpClass(self):
        """Setup inputs needed for tests in the class."""

        self.ySmall = numpy.array([[1, 0], [0, 1], [1, 0]])

    def test_pass_small(self):
        partition = PLS.partition_dataset.partition_dataset(self.ySmall, 2)

    def test_too_many_folds(self):
        partition = PLS.partition_dataset.partition_dataset(self.ySmall, 2)

    # Tests to add
    # CV stratified and non-stratified
    # Larger matrices


class CorrectnessTests(unittest.TestCase):
    """Tests checking the correctness of the output."""

    pass


if __name__ == '__main__':
    unittest.main()