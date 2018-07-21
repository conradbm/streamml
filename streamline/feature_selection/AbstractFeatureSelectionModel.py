class AbstractFeatureSelectionModel:
    
    #properties
    _model=None
    _grid=None
    _pipe=None
    _params=None
    _modelType=None
    _validator=None
    _validation_results=None
    _X=None
    _y=None
    _code=None
    _n_jobs=None
    _verbose=None
    _features=None
    
    def __init__(self):
        pass