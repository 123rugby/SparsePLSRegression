import numpy
import PLS.center_and_store
from scipy import sparse
import unittest


class CompletionTests(unittest.TestCase):
    """Tests checking whether the file based dot product code successfully returns."""

    @classmethod
    def setUpClass(cls):
        """Setup inputs needed for tests in the class."""

        pass


class CorrectnessTests(unittest.TestCase):
    """Tests checking the correctness of the output."""

    pass

    # These will just be doing dot procuts of the (centerd) matrices in numpy, and then seeing whether saving
    # and dot producting from the file gives te same answer (with some tolerance for loss of precision due to storage)


if __name__ == '__main__':
    unittest.main()