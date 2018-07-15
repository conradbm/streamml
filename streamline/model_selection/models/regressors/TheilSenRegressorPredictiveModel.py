import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from streamline.model_selection.models.AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.linear_model import TheilSenRegressor

class TheilSenRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, tsr_params, nfolds=3, n_jobs=1, scoring=None, verbose=True):
        
        self._code="tsr"
        
        if verbose:
            print ("Constructed AdaptiveBoostingRegressorPredictiveModel: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, tsr_params, nfolds, n_jobs, scoring, verbose)
        self._model = self.constructRegressor(TheilSenRegressor())
        
    
    #methods
    def execute(self):
        pass
       