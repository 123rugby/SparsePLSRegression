import numpy
from scipy import sparse

def pls(X, Y, numberComponents=10, convergeThreshold=1e-9, maxIterations=1000):
    """Perform PLS regression using the NIPALS algorithm.

    :param X:                   The n x p matrix of predictors.
    :type X:                    scipy.sparse matrix (ideally CSC, but any form for fast computations will work)
    :param Y:                   The n x m matrix of responses.
    :type Y:                    numpy.array
    :param numberComponents:    The number of components (latent variables) to use.
    :type numberComponents:     int
    :param convergeThreshold:   The threshold to use to determine whether the NIPALS algorithm has converged for the component.
    :type convergeThreshold:    float
    :param maxIterations:       The maximum number of NIPALS iterations to perform per component.
    :type maxIterations:        int
    :returns :                  ....
    :type :                     ....

    """
    
    # Determine dimensions of inputs.
    [numObservations, numPredictors] = X.shape
    [numObservations, numResponses] = Y.shape
    
    print("Obs: {0:d}, Predictors: {1:d}, Responses: {2:d}".format(numObservations, numPredictors, numResponses))
    
    # Initialise outputs.
    xLoading = numpy.zeros((numPredictors, numberComponents))
        # Each row contains coefficients that define a linear combination of the components that approximate the original predictor variables.
        # The coefficients from regressing centred X, which we'll call X0, on the x scores: XL = (XS\X0)' = X0'*XS.
        # XS*XL' is the PLS approximation to X0.
    xScore = numpy.zeros((numObservations, numberComponents))
        # The components that are linear combinations of the variables in X.
    yLoading = numpy.zeros((numResponses, numberComponents))
        # Each row contains coefficients that define a linear combination of PLS components that approximate the original response variables.
        # The coefficients from regressing centred Y, which we'll call Y0, on the x scores: YL = (XS\Y0)' = Y0'*XS.
        # XS*YL' is the PLS approximation to Y0.
    yScore = numpy.zeros((numObservations, numberComponents))
        # The linear combinations of the responses with which the components xScore have maximum covariance.
    weights = numpy.zeros((numPredictors, numberComponents))
        # A p-by-ncomp matrix of PLS weights W so that XS = X0*W.
    
    # Calculate the outputs for each component.
    for i in range(numberComponents):
        distance = 1  # Normalised distance between the X matrix's score for this component at iteration i and i - 1.
        xScore = Y[:, 0]  # Initialise the X matrix's score for this component to the first column of the target matrix Y.
        numIterations = 0  # Number of iterations performed.
        while ((distance > convergeThreshold) and (numIterations < maxIterations)):
            # Recalculate while convergence has not been reached and there are iterations still available.
            xLoading = (X.T).dot(xScore) / ((xScore.T).dot(xScore))  # Project X onto its score for this component to get its loading.????
            xLoading = xLoading / numpy.linalg.norm(xLoading)  # Normalise the loading.
            yScore = X.dot(xLoading)  # Project X onto this component to get the Y score.????
            yLoading = (Y.T).dot(yScore) / ((yScore.T).dot(yScore))  # Project Y onto its score for this component to get its loading.????
            xScoreNext = Y.dot(yLoading) / ((yLoading.T).dot(yLoading))  # ????
            distance = numpy.linalg.norm(xScoreNext - xScore) / numpy.linalg.norm(xScore)  # Determine the distance between the new X score and the old.
            xScore = xScoreNext
            numIterations += 1
        p = (X.T).dot(yScore) / ((yScore.T).dot(yScore))  # p=X'*t/(t'*t);
        
        
        p=X'*t/(t'*t);
        X=X-t*p';
        Y=Y-t*q';

        %+++ store
        W(:,i)=w;
        T(:,i)=t;
        P(:,i)=p;
        Q(:,i)=q;

    Wstar=W*(P'*W)^(-1);
    B=Wstar*Q';
    Q=Q';

        # 

    print(xScore.shape)  # numObs entries - prob X scores
    print(xLoading.shape)  # numPred entries - prob X loading
    
    print(yScore.shape)  # numObs entries - prob Y scores
    print(yLoading.shape)  # numResp entries - prob Y loadings
    print(p.shape)  # numPred entries
    
    # Regress X onto the scores and get a loading factor
    # Regress Y against/on X: E(Y)=aX+b
    #       ignoring b you then get Y/X = a
    #
    # Should i say project X/Y onto its score to get the loading, or should I say regress X/Y onto its score?
    #
    # Any line with ???? in the comment really needs checking
    
    # Return B (coef), Wstar (weights), X scores, X loadings, Y scores, Y loadings