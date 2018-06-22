import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
import matplotlib.pyplot as plt
import sys
import os
#sys.path.append(os.path.abspath(sys.path[0]+"/src/streamline/model_selection/models/regressors/"))

from streamml.streamline.model_selection.models.regressors.LinearRegressorPredictiveModel import LinearRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.SupportVectorRegressorPredictiveModel import SupportVectorRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.RidgeRegressorPredictiveModel import RidgeRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.LassoRegressorPredictiveModel import LassoRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.ElasticNetRegressorPredictiveModel import ElasticNetRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.KNNRegressorPredictiveModel import KNNRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.RandomForestRegressorPredictiveModel import RandomForestRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.AdaptiveBoostingRegressorPredictiveModel import AdaptiveBoostingRegressorPredictiveModel


"""
Example Usage:

import pandas as pd
import numpy as np
from streamml.streamline.transformation.TransformationStream import TransformationStream

X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))
y = pd.DataFrame(np.array([np.random.exponential() for i in range(200)]))

#options: lr, ridge, lasso, enet, svr, knnr, abr, rfr
RES = ModelSelectionStream(Xnew,y).flow(["svr", "lr", "knnr","lasso","abr"],
                                              params={'svr__C':[1,0.1,0.01,0.001],
                                                     'lr__fit_intercept':[False, True],
                                                     'knnr__n_neighbors':[5,10],
                                                     'lasso__alpha':[1.0,10.0,20.0],
                                                     'abr__n_estimators':[10,20,50],
                                                     'abr__learning_rate':[0.1,1,10]},
                                              verbose=True)
"""

class ModelSelectionStream:
    #properties
    _X=None
    _y=None
    _test_size=None
    _nfolds=None
    _n_jobs=None
    _verbose=None
    _scoring=None
    _metrics=None
    _test_size=None
    _wrapper_models=None
    _bestEstimators={}
    _bestEstimator=None
    _regressors_results=None
    _classifiers_results=None
    
    """
    Constructor:
    1. Default
        Paramters: df : pd.DataFrame, dataframe must be accepted to use this class
    """
    def __init__(self,X,y):
        assert isinstance(X, pd.DataFrame), "X was not a pandas DataFrame"
        assert any([isinstance(y,pd.DataFrame), isinstance(y,pd.Series)]), "y was not a pandas DataFrame or Series"
        self._X = X
        self._y = y
       
    """
	Methods:
	getBestEstimators
	"""
    def getBestEstimators(self):
        return self._bestEstimators

    """
	Methods:
	getBestEstimator
	"""
    def getBestEstimator(self):
        return self._bestEstiminator

	"""
	Methods:
	determineBestEstimators
	"""
    def determineBestEstimators(self, models):
        if self._verbose:
            print("**************************************************")
            print("Determining Best Estimators.")
            print("**************************************************")
        for model in models:
            self._bestEstimators[model.getCode()]=model.getBestEstimator()

            if self._verbose:
                print(model.getCode(), model.getBestEstimator().get_params())
        return self._bestEstimators

	"""
	Methods:
	handleRegressors
	"""
    def handleRegressors(self, Xtest, ytest, metrics, wrapper_models):
        

        self._regressors_results={}
        for model in wrapper_models:
            self._regressors_results[model.getCode()]=model.validate(Xtest, ytest, metrics)
        
        # create a pandas dataframe of each metric on each model
        
        if self._verbose:
            print("**************************************************")
            print("Regressor Performance Sheet")
            print("**************************************************")
            
            df = pd.DataFrame(self._regressors_results)
            print(df)
            df.plot(kind='line', title='Errors by Model')
            plt.show()
            # plot models against one another in charts
        
        
        return self._regressors_results
    
	"""
	Methods:
	handleClassifiers
	"""
    def handleClassifiers(self, Xtest, ytest, metrics, wrapper_models):
        if self._verbose:
            print("**************************************************")
            print("Classifier Performance Sheet")
            print("**************************************************")
        pass
    
	"""
	Methods:
	handleModelSelection
	"""
    def handleModelSelection(self, regressors, metrics, Xtest, ytest, wrapper_models):
        
        if regressors:
            self._bestEstimator = self.handleRegressors(Xtest, ytest, metrics, wrapper_models)
        else:
            #classifiers
            assert 1 == 2, "Handling Classification is not yet supported."
            self._bestEstimator = self.handleClassifiers(Xtest, ytest, metrics, wrapper_models)
            
        return self._bestEstimator
    

    """
    Methods:
    1. flow
        Parameters: models_to_flow : list(), User specified models to optimize and compare against one another. 

                    MetaParamters:
                        - lr -> LinearRegression()
                        - ridge -> Ridge()
                        - lasso -> Lasso()
                        - enet -> ElasticNet()
                        - svr -> SVR()
                        - knnr -> KNearestRegression()
                        - abr -> AdaptiveBoostingRegression()
                        - rfr -> RandomForestRegression()
                    
    """
    def flow(self, 
             models_to_flow=[], 
             params=None, 
             test_size=0.2, 
             nfolds=3, 
             nrepeats=3,
             pos_split=1,
             n_jobs=1, 
             metrics=[], 
             verbose=False, 
             regressors=True):
      
        assert isinstance(nfolds, int), "nfolds must be integer"
        assert isinstance(nrepeats, int), "nrepeats must be integer"
        assert isinstance(n_jobs, int), "n_jobs must be integer"
        assert isinstance(verbose, bool), "verbosem ust be bool"
        assert isinstance(pos_split, int), "pos_split must be integer"
        assert isinstance(params, dict), "params must be a dict"
        assert isinstance(test_size, float), "test_size must be a float"
        assert isinstance(metrics, list), "model scoring must be a list"
        assert isinstance(regressors, bool), "regressor must be bool"

        self._nfolds=nfolds
        self._nrepeats=nrepeats
        self._n_jobs=n_jobs
        self._verbose=verbose
        self._pos_split=pos_split
        self._allParams=params
        self._metrics=metrics
        self._test_size=test_size
        self._regressors=regressors
        
        # Inform the streamline to user.
        stringbuilder=""
        for thing in models_to_flow:
            stringbuilder += thing
            stringbuilder += " --> "
            
        if self._verbose:
            print("**************************************************")
            print("Model Selection Streamline: " + stringbuilder[:-5])
            print("**************************************************")
        
        def linearRegression():
            
            self._lr_params={}
            for k,v in self._allParams.items():
                if "lr" in k:
                    self._lr_params[k]=v

                
            model = LinearRegressorPredictiveModel(self._X_train, 
                                                   self._y_train,
                                                   self._lr_params,
                                                   self._nfolds, 
                                                   self._n_jobs,
                                                   self._verbose)
            return model
            
        def supportVectorRegression():
            self._svr_params={}
            for k,v in self._allParams.items():
                if "svr" in k:
                    self._svr_params[k]=v

                
            model = SupportVectorRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._svr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        
        def randomForestRegression():
            self._rfr_params={}
            for k,v in self._allParams.items():
                if "rfr" in k:
                    self._rfr_params[k]=v

                
            model = RandomForestRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._rfr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        

        
        def adaptiveBoostingRegression():
            self._abr_params={}
            for k,v in self._allParams.items():
                if "abr" in k:
                    self._abr_params[k]=v

                
            model = AdaptiveBoostingRegressorPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._abr_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model
            
        def knnRegression():
            self._knnr_params={}
            for k,v in self._allParams.items():
                if "knnr" in k:
                    self._knnr_params[k]=v

            
            
            model = KNNRegressorPredictiveModel(self._X_train, 
                                                self._y_train,
                                                self._knnr_params,
                                                self._nfolds, 
                                                self._n_jobs,
                                                self._verbose)
            
            return model
            
        def ridgeRegression():
            self._ridge_params={}
            for k,v in self._allParams.items():
                if "ridge" in k:
                    self._ridge_params[k]=v

                
            model = RidgeRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._ridge_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        
        def lassoRegression():
            self._lasso_params={}
            for k,v in self._allParams.items():
                if "lasso" in k:
                    self._lasso_params[k]=v

                
            model = LassoRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._lasso_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        
        def elasticNetRegression():
            self._enet_params={}
            for k,v in self._allParams.items():
                if "enet" in k:
                    self._enet_params[k]=v

            model = ElasticNetRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._enet_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model
            
        
        #options: lr, ridge, lasso, enet, svr, knnr, abr, rfr
        # Define our model selection options
        options = {"lr" : linearRegression,
                   "svr" : supportVectorRegression,
                   "rfr":randomForestRegression,
                   "abr":adaptiveBoostingRegression,
                   "knnr":knnRegression,
                   "ridge":ridgeRegression,
                   "lasso":lassoRegression,
                   "enet":elasticNetRegression}
        
        
		# Define our training and test sets
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                     self._y,
                                                                                     test_size=self._test_size)
        # Confirm training test splits worked
		#print(self._X_train.shape)
        #print(self._X_test.shape)
        #print(self._y_train.shape)
        #print(self._y_test.shape)
            
        # Accumulate each wrapper model the user wants to execute on
        self._wrapper_models=[]
        for key in models_to_flow:
             self._wrapper_models.append(options[key]())
        
        if self._verbose:
            print
		# Execute the users request on wrapper models
        self._bestEstimators = self.determineBestEstimators(self._wrapper_models)
        
        
		# If metrics defined, tell the user which model did best with a visualization
        if len(self._metrics) > 0:
            self._bestEstiminator = self.handleModelSelection(self._regressors, 
                                                              self._metrics, 
                                                              self._X_test, 
                                                              self._y_test, 
                                                              self._wrapper_models)

        
		# Return each best estimator the user is interested in
        return self._bestEstimators
    

