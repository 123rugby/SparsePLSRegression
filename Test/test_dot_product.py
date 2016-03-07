import numpy
import os
import PLS.dot_product
import unittest


class DotProductTests(unittest.TestCase):
    """Tests to determine whether the file-based dot product returns acceptable results."""

    def test_correctness(self):
        """Test whether the file and memory dot products return the same values."""

        # Generate the matrices to test.
        matrices = []
        matrices.append(numpy.random.rand(4, 5))
        matrices.append(numpy.random.rand(5, 4))
        matrices.append(numpy.random.rand(5, 5))
        matrices.append(numpy.random.rand(10, 10))
        matrices.append(numpy.random.rand(20, 20))
        matrices.append(numpy.random.rand(50, 50))
        matrices.append(numpy.random.rand(100, 100))
        matrices.append(numpy.random.rand(1000, 1000))

        # Test the matrices.
        fileMatrix = "TestMatrixDot.txt"
        fileMatrixTrans = "TestMatrixTransDot.txt"
        comparisons = []
        for i in matrices:
            # Save the matrix (both transpose and not).
            numpy.savetxt(fileMatrix, i, delimiter='\t')
            numpy.savetxt(fileMatrixTrans, i.T, delimiter='\t')

            # Calculate the dot products.
            [numRows, numCols] = i.shape
            memDotProd = i.dot(i.T)
            fileDotProd = PLS.dot_product.dot_product(fileMatrix, numRows, i.T, sep='\t')
            memDotProdTrans = (i.T).dot(i)
            fileDotProdTrans = PLS.dot_product.dot_product(fileMatrixTrans, numCols, i, sep='\t')

            # Determine whether the products are equivalent to 10 decimal places
            # (and therefore basically the tolerance of float arithmetic).
            comparisons.append(numpy.allclose(memDotProd, fileDotProd, rtol=0, atol=1e-10))
            comparisons.append(numpy.allclose(memDotProdTrans, fileDotProdTrans, rtol=0, atol=1e-10))

        # Remove temporary files used.
        os.remove(fileMatrix)
        os.remove(fileMatrixTrans)

        # Output result.
        self.assertTrue(all(comparisons))


if __name__ == '__main__':
    unittest.main()