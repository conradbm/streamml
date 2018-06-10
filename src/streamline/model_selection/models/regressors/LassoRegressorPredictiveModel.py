import sys
import os
sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/"))
from AbstractRegressorPredictiveModel import AbstractRegressorPredictiveModel

from sklearn.linear_model import Lasso
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class LassoRegressorPredictiveModel(AbstractRegressorPredictiveModel):
    #properties

    #constructor
    def __init__(self, X, y, lr_params, nfolds=3, n_jobs=2, verbose=True):
        
        self._nfolds=nfolds
        self._n_jobs=n_jobs
        self._verbose=verbose
        self._code="lasso"
        
        if verbose:
            print ("Constructed LassoRegressorPredictiveModel: " +self._code)
        
        AbstractRegressorPredictiveModel.__init__(self, "regressor", X, y, lr_params)
        self._model = self.constructRegressor()
        
    
    #methods
    def execute(self):
        pass
    
    def constructRegressor(self):
        self._pipe          = Pipeline([(self._code, Lasso())])

        self._grid          = GridSearchCV(self._pipe,
                                           param_grid=self._params, 
                                           n_jobs=self._n_jobs, 
                                           cv=self._nfolds, 
                                           verbose=False)

        best_fit                 = self._grid.fit(self._X,self._y).best_estimator_.named_steps[self._code]
        return best_fit 