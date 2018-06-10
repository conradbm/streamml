
class AbstractPredictiveModel:
    #properties
    _model=None
    _grid=None
    _pipe=None
    _params=None
    _modelType=None
    _validator=None
    _X=None
    _y=None
    _code=None
    _n_jobs=None
    _verbose=None
    
    #constructor
    def __init__(self, X, params):
        if self._verbose:
            print ("Constructed AbstractPredictiveModel: "+self._code)
        assert isinstance(params, dict), "params must be dict"
        self._X = X
        self._params = params
        self._validator = ModelValidation()
    #methods
    def execute(self):
        pass
    
    def getCode(self):
        return self._code
    
    def getBestEstimator(self):
        if self._verbose:
            print("Returning "+self._code+" best estiminator")
        return self._model