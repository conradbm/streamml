import sys
import os
from streamline.model_selection.models.AbstractClassifierPredictiveModel import AbstractClassifierPredictiveModel

from sklearn..neighbors import KNeighborsClassifier

class KNeighborsClassifierPredictiveModel(AbstractClassifierPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, knnc_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="knnc"
        
        if verbose:
            print ("Constructed KNeighborsClassifierPredictiveModel: " +self._code)
        
        AbstractClassifierPredictiveModel.__init__(self, "classifier", X, y, knnc_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructClassifier(KNeighborsClassifier())
        
       
