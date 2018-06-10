import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import sys
import os
sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/regressors/"))
from LinearRegressorPredictiveModel import LinearRegressorPredictiveModel
from SupportVectorRegressorPredictiveModel import SupportVectorRegressorPredictiveModel
from RidgeRegressorPredictiveModel import RidgeRegressorPredictiveModel
from LassoRegressorPredictiveModel import LassoRegressorPredictiveModel
from ElasticNetRegressorPredictiveModel import ElasticNetRegressorPredictiveModel
from KNNRegressorPredictiveModel import KNNRegressorPredictiveModel
from RandomForestRegressorPredictiveModel import RandomForestRegressorPredictiveModel
from AdaptiveBoostingRegressorPredictiveModel import AdaptiveBoostingRegressorPredictiveModel

class ModelSelectionStream:
    #properties
    _X=None
    _y=None
    _test_size=None
    _nfolds=None
    _n_jobs=None
    _verbose=None
    
    #constructor
    def __init__(self,X,y):
        assert isinstance(X, pd.DataFrame), "X was not a pandas DataFrame"
        assert any([isinstance(y,pd.DataFrame), isinstance(y,pd.Series)]), "y was not a pandas DataFrame or Series"
        self._X = X
        self._y = y
        
    
    #methods
    def flow(self, models_to_flow=[], params=None, test_size=0.2, nfolds=3, nrepeats=3, pos_split=1, n_jobs=2, verbose=False):
        
        assert isinstance(nfolds, int), "nfolds must be integer"
        assert isinstance(nrepeats, int), "nrepeats must be integer"
        assert isinstance(n_jobs, int), "n_jobs must be integer"
        assert isinstance(verbose, bool), "verbosem ust be bool"
        assert isinstance(pos_split, int), "pos_split must be integer"
        assert isinstance(params, dict), "params must be a dict"
        
        self._nfolds=nfolds
        self._nrepeats=nrepeats
        self._n_jobs=n_jobs
        self._verbose=verbose
        self._pos_split=pos_split
        self._allParams=params
        
        # Inform the streamline to user.
        stringbuilder=""
        for thing in models_to_flow:
            stringbuilder += thing
            stringbuilder += " --> "
        print("Model Selection Streamline: " + stringbuilder[:-5])
    
        
        def linearRegression():
            
            self._lr_params={}
            for k,v in self._allParams.items():
                if "lr" in k:
                    self._lr_params[k]=v
            
            if self._verbose:
                print("Executing Linear Regressor")
                
            model = LinearRegressorPredictiveModel(self._X_train, 
                                                   self._y_train,
                                                   self._lr_params,
                                                   self._nfolds, 
                                                   self._n_jobs,
                                                   self._verbose)
            return model.getBestEstimator()
            
        def supportVectorRegression():
            self._svr_params={}
            for k,v in self._allParams.items():
                if "svr" in k:
                    self._svr_params[k]=v
            
            if self._verbose:
                print("Executing Support Vector Regressor")
                
            model = SupportVectorRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._svr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model.getBestEstimator()
            
        
        def randomForestRegression():
            self._rfr_params={}
            for k,v in self._allParams.items():
                if "rfr" in k:
                    self._rfr_params[k]=v
            
            if self._verbose:
                print("Executing Random Forest Regressor")
                
            model = RandomForestRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._rfr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model.getBestEstimator()
            
        

        
        def adaptiveBoostingRegression():
            self._abr_params={}
            for k,v in self._allParams.items():
                if "abr" in k:
                    self._abr_params[k]=v
            
            if self._verbose:
                print("Executing Adaptive Boosting Regressor")
                
            model = AdaptiveBoostingRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._abr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model.getBestEstimator()
            
        def knnRegression():
            self._knnr_params={}
            for k,v in self._allParams.items():
                if "knnr" in k:
                    self._knnr_params[k]=v
            
            if self._verbose:
                print("Executing K-Nearest Neighbors Regressor")
            
            
            model = KNNRegressorPredictiveModel(self._X_train, 
                                                self._y_train,
                                                self._knnr_params,
                                                self._nfolds, 
                                                self._n_jobs,
                                                self._verbose)
            
            return model.getBestEstimator()
            
        def ridgeRegression():
            self._ridge_params={}
            for k,v in self._allParams.items():
                if "ridge" in k:
                    self._ridge_params[k]=v
            
            if self._verbose:
                print("Executing Ridge Regressor")
                
            model = RidgeRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._ridge_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model.getBestEstimator()
            
        
        def lassoRegression():
            self._lasso_params={}
            for k,v in self._allParams.items():
                if "lasso" in k:
                    self._lasso_params[k]=v
            
            if self._verbose:
                print("Executing Lasso Regressor")
                
            model = LassoRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._lasso_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model.getBestEstimator()
            
        
        def elasticNetRegression():
            self._enet_params={}
            for k,v in self._allParams.items():
                if "enet" in k:
                    self._enet_params[k]=v
            
            if self._verbose:
                print("Executing Elastic Net Regressor")
                
            model = ElasticNetRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._enet_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model.getBestEstimator()
            
        
        #options: lr, ridge, lasso, enet, svr, knnr, abr, rfr
        # map the inputs to the function blocks
        options = {"lr" : linearRegression,
                   "svr" : supportVectorRegression,
                   "rfr":randomForestRegression,
                   "abr":adaptiveBoostingRegression,
                   "knnr":knnRegression,
                   "ridge":ridgeRegression,
                   "lasso":lassoRegression,
                   "enet":elasticNetRegression}
        
        
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                     self._y,
                                                                                     test_size=0.2)
        print(self._X_train.shape)
        print(self._X_test.shape)
        print(self._y_train.shape)
        print(self._y_test.shape)
            
        # Execute commands as provided in the preproc_args list
        models=[]
        for key in models_to_flow:
             models.append(options[key]())
        
        for model in models:
            
            
        pass
        