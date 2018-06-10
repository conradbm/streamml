import sys
import os
sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/"))
print(sys.path)
from AbstractPredictiveModel import AbstractPredictiveModel

class AbstractRegressorPredictiveModel(AbstractPredictiveModel):

    #constructor

    def __init__(self, modelType, X, y, params):
        
        if self._verbose:
            print("Constructed AbstractRegressorPredictiveModel: "+self._code)
        assert modelType == "regressor", "You are creating a regressor, but have no specified it to be one."
        
        #assert isinstance(y.dtypes,float), "Your response variable y is not a float."
        self._modelType = modelType
        self._y=y
        AbstractPredictiveModel.__init__(self, X, params)
        
    #methods
    def execute(self):
        pass
    
    def constructRegressor(self):
        pass