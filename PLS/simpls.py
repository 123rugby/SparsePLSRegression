import numpy

def simpls(X, Y, numberComponents=10):
    """Run the standard SIMPLS algorithm.

    :param X:                   The n x p matrix of predictors.
    :type X:                    scipy.sparse matrix (ideally CSC, but any form for fast computations will work)
    :param Y:                   The n x m matrix of responses.
    :type Y:                    numpy.array
    :param numberComponents:    The number of components (latent variables) to use.
    :type numberComponents:     int
    :returns :                  ....
    :rtype :                     ....

    """
    
    # Determine dimensions of inputs.
    [numObservations, numPredictors] = X.shape
    yDimensions = Y.shape
    if len(yDimensions) == 1:
        # There is only one response variable (PLS1).
        numObservationsY = Y.shape[0]
        numResponses = 1
        Y = Y.reshape(numObservationsY, 1)  # Ensure that Y is a column vector.
    else:
        # There are multiple response variables (PLS2).
        [numObservationsY, numResponses] = yDimensions
    
    # Initialise outputs.
    xLoadings = numpy.matrix(numpy.zeros((numPredictors, numberComponents)))
        # Each row contains coefficients that define a linear combination of the components that approximate the original predictor variables.
        # The coefficients from regressing centred X, which we'll call X0, on the x scores: XL = (XS\X0)' = X0'*XS.
        # XS*XL' is the PLS approximation to X0.
    xScores = numpy.matrix(numpy.zeros((numObservations, numberComponents)))
        # The components that are linear combinations of the variables in X.
    yLoadings = numpy.matrix(numpy.zeros((numResponses, numberComponents)))
        # Each row contains coefficients that define a linear combination of PLS components that approximate the original response variables.
        # The coefficients from regressing centred Y, which we'll call Y0, on the x scores: YL = (XS\Y0)' = Y0'*XS.
        # XS*YL' is the PLS approximation to Y0.
    yScores = numpy.matrix(numpy.zeros((numObservations, numberComponents)))
        # The linear combinations of the responses with which the components xScore have maximum covariance.
    weights = numpy.matrix(numpy.zeros((numPredictors, numberComponents)))
        # A p-by-ncomp matrix of PLS weights W so that XS = X0*W.

    # An orthonormal basis for the span of the X loadings, to make the successive deflation X0'*Y0 simple.
    # Each new basis vector can be removed from Cov separately.
    V = numpy.matrix(numpy.zeros((numPredictors, numberComponents)))

    Cov = numpy.matrix((X.T).dot(Y))
    for i in range(numberComponents):
        # Find unit length ti=X0*ri and ui=Y0*ci whose covariance, ri'*X0'*Y0*ci, is
        # jointly maximized, subject to ti'*tj=0 for j=1:(i-1).
        [R, S, C] = numpy.linalg.svd(Cov)
        r = numpy.matrix(R)[:, 0]
        c = numpy.matrix(C)[:, 0]
        s = S[0]  # First component
        t = X.dot(r)
        normT = numpy.linalg.norm(t)
        t = t / normT  # t' * t = 1        
        xLoadings[:, i] = (X.T).dot(t)
        q = (s * c) / normT  # = Y0'*ti
        yLoadings[:, i] = q
        xScores[:, i] = t
        yScores[:, i] = Y.dot(q)  # = Y0*(Y0'*ti), and proportional to Y0*ci
        weights[:, i] = r / normT  # rescaled to make ri'*X0'*X0*ri == ti'*ti == 1

        # Update the orthonormal basis with modified Gram Schmidt (more stable),
        # repeated twice (ditto).
        v = xLoadings[:, i]
        for j in range(2):
            for k in range(i):
                vj = V[:, j]
                v = v - numpy.multiply((vj.T).dot(v), vj)
        v = v / numpy.linalg.norm(v)
        V[:, i] = v

        # Deflate Cov, i.e. project onto the ortho-complement of the X loadings.
        # First remove projections along the current basis vector, then remove any
        # component along previous basis vectors that's crept in as noise from
        # previous deflations.
        Cov = Cov - numpy.multiply(v, (v.T).dot(Cov))
        Vi = V[:, 0:i+1]
        Cov = Cov - Vi.dot((Vi.T).dot(Cov))

    # By convention, orthogonalize the Y scores w.r.t. the preceding Xscores,
    # i.e. XSCORES'*YSCORES will be lower triangular.  This gives, in effect, only
    # the "new" contribution to the Y scores for each PLS component.  It is also
    # consistent with the PLS-1/PLS-2 algorithms, where the Y scores are computed
    # as linear combinations of a successively-deflated Y0.  Use modified
    # Gram-Schmidt, repeated twice.
    for i in range(numberComponents):
        u = yScores[:, i]
        for j in range(2):
            for k in range(i):
                tj = xScores[:, k]
                u = u - numpy.multiply((tj.T).dot(u), tj)
        yScores[:, i] = u

    return xLoadings, yLoadings, xScores, yScores, weights