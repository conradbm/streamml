import sys
import os
from ..AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn.linear_model import LogisticRegression

class LogisticRegressionClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, logr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="logr"
        
        if verbose:
            print ("Constructed LogisticRegressionClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, logr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructClassifier(LogisticRegression())
        
       
