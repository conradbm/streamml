
# Data manipulation
import pandas as pd
import numpy as np

# Statistics
from statsmodels.regression import linear_model
import statsmodels.api as sm

# Data Splitting
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold

# Plotting
import matplotlib.pyplot as plt
import seaborn as sns

# System
import sys
import os

# Data containers
from collections import defaultdict

# Print Settings
import warnings
warnings.filterwarnings("ignore")

# Regressors
from streamml.streamline.model_selection.models.regressors.SupportVectorRegressorPredictiveModel import SupportVectorRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.LassoRegressorPredictiveModel import LassoRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.ElasticNetRegressorPredictiveModel import ElasticNetRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.RandomForestRegressorPredictiveModel import RandomForestRegressorPredictiveModel
from streamml.streamline.model_selection.models.regressors.AdaptiveBoostingRegressorPredictiveModel import AdaptiveBoostingRegressorPredictiveModel

# X
#from streamml.streamline.feature_selection.models.regressors.MixedSelectionRegressionFeatureSelectionModel import MixedSelectionRegressionFeatureSelectionModel

# X
#from streamml.streamline.feature_selection.models.regressors.PartialLeastSquaresRegressionFeatureSelectionModel import PartialLeastSquaresRegressionFeatureSelectionModel


# Classifiers
from streamml.streamline.model_selection.models.classifiers.AdaptiveBoostingClassifierPredictiveModel import AdaptiveBoostingClassifierPredictiveModel
from streamml.streamline.model_selection.models.classifiers.RandomForestClassifierPredictiveModel import RandomForestClassifierPredictiveModel

# Ensembler
# X
#from streamml.streamline.feature_selection.ensemble.TOPSISEnsembleFeatureSelectionModel import TOPSISEnsembleFeatureSelectionModel


class FeatureSelectionStream:
        #properties
    _X=None
    _y=None
    _test_size=None
    _nfolds=None
    _n_jobs=None
    _verbose=None
    _metrics=None
    _test_size=None
    _wrapper_models=None
    _bestEstimators=None
    _regressors_results=None
    _classifiers_results=None
    _modelSelection=None
    
    """
    Constructor: __init__:
    
    @param: X : pd.DataFrame, dataframe representing core dataset.
    @param: y : pd.DataFrame, dataframe representing response variable, either numeric or categorical.
    """
    def __init__(self,X,y):
        assert isinstance(X, pd.DataFrame), "X was not a pandas DataFrame"
        assert any([isinstance(y,pd.DataFrame), isinstance(y,pd.Series)]), "y was not a pandas DataFrame or Series"
        self._X = X
        self._y = y

    
    """
    Methods: flow
    
    @usage meant to make models flow
    
    @param models_to_flow, list.
    @param params, dict.
                  
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
             regressors=True,
             ensemble=False,
             cut=None):
      
        assert isinstance(nfolds, int), "nfolds must be integer"
        assert isinstance(nrepeats, int), "nrepeats must be integer"
        assert isinstance(n_jobs, int), "n_jobs must be integer"
        assert isinstance(verbose, bool), "verbosem ust be bool"
        assert isinstance(pos_split, int), "pos_split must be integer"
        assert isinstance(params, dict), "params must be a dict"
        assert isinstance(test_size, float), "test_size must be a float"
        assert isinstance(metrics, list), "model scoring must be a list"
        assert isinstance(regressors, bool), "regressor must be bool"
        assert isinstance(ensemble, bool), "ensemble must be bool"
        
        
        # Mixed Selection Parameters
            # if defined, then set
            # else
                #self._initial_list=[]
                #self._threshold_in=0.01
                #self._threshold_out = 0.05
            
        
        self._nfolds=nfolds
        self._nrepeats=nrepeats
        self._n_jobs=n_jobs
        self._verbose=verbose
        self._pos_split=pos_split
        self._allParams=params
        self._metrics=metrics
        self._test_size=test_size
        self._regressors=regressors
        self._ensemble=ensemble
        self._cut = cut
        
        # Inform the streamline to user.
        stringbuilder=""
        for thing in models_to_flow:
            stringbuilder += thing
            stringbuilder += " --> "
            
        if self._verbose:
            
            if self._regressors:
                print("*************************")
                print("=> (Regressor) "+"=> Feature Selection Streamline: " + stringbuilder[:-5])
                print("*************************")
            elif self._regressors == False:
                print("*************************")
                print("=> (Classifier) "+"=> Feature Selection Streamline: " + stringbuilder[:-5])
                print("*************************")
            else:
                print("Invalid model selected. Please set regressors=True or regressors=False.")
                print
                
                
                
                
    
        def supportVectorRegression():
            self._svr_params={}
            for k,v in self._allParams.items():
                if "svr" in k:
                    self._svr_params[k]=v

            
            self._svr_params["kernel"]='linear'
            model = SupportVectorRegressorPredictiveModel(self._X_train, 
                                                          self._y_train,
                                                          self._svr_params,
                                                          self._nfolds, 
                                                          self._n_jobs,
                                                          self._verbose)
            return model.getBestEstimator().coef_
            
        
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
            return model.getBestEstimator().feature_importances_
            
        

        
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
            return model.getBestEstimator().feature_importances_
        
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
            return model.getBestEstimator().coef_
            
        
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
            return model.getBestEstimator().coef_
    
        def mixed_selection():
            
            if self._verbose:
                print("Executing: mixed_selection")
            
            
            X = self._X
            y = self._y
            
            initial_list=[]
            threshold_in=0.01
            threshold_out = 0.05
            verbose=True
            
            #initial_list = self._initial_list
            #threshold_in = self._threshold_in
            #threshold_out = self._threshold_out
            #verbse = self._verbose
            
            """ Perform a forward-backward feature selection 
            based on p-value from statsmodels.api.OLS
            Arguments:
                X - pandas.DataFrame with candidate features
                y - list-like with the target
                initial_list - list of features to start with (column names of X)
                threshold_in - include a feature if its p-value < threshold_in
                threshold_out - exclude a feature if its p-value > threshold_out
                verbose - whether to print the sequence of inclusions and exclusions
            Returns: list of selected features 
            Always set threshold_in < threshold_out to avoid infinite looping.
            See https://en.wikipedia.org/wiki/Stepwise_regression for the details
            """
                      
            included = list(initial_list)
            while True:
                changed=False

                # forward step
                excluded = list(set(X.columns)-set(included))
                new_pval = pd.Series(index=excluded)
                for new_column in excluded:
                    model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included+[new_column]]))).fit()
                    new_pval[new_column] = model.pvalues[new_column]

                best_pval = new_pval.min()



                if best_pval < threshold_in:
                    best_feature = new_pval.idxmin()
                    #best_feature = new_pval.argmin()
                    included.append(best_feature)
                    changed=True
                    if verbose:
                        print('Add  {:30} with p-value {:.6}'.format(best_feature, best_pval))

                # backward step
                model = sm.OLS(y, sm.add_constant(pd.DataFrame(X[included]))).fit()
                # use all coefs except intercept
                pvalues = model.pvalues.iloc[1:]
                worst_pval = pvalues.max() # null if pvalues is empty
                if worst_pval > threshold_out:
                    changed=True
                    worst_feature = pvalues.idxmax()
                    #worst_feature = pvalues.argmax()
                    included.remove(worst_feature)
                    if verbose:
                        print('Drop {:30} with p-value {:.6}'.format(worst_feature, worst_pval))

                if not changed:
                    break


            return included

        
        
        
        def partialLeastSquaresRegression():
            pass
    
        ############################################
        ########## Classifiers Start Here ##########
        ############################################
        
        def adaptiveBoostingClassifier():
            self._abc_params={}
            for k,v in self._allParams.items():
                if "abc" in k:
                    self._abc_params[k]=v

                
            model = AdaptiveBoostingClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._abc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model.getBestEstimator().feature_importances_
        
        def randomForestClassifier():
            self._rfc_params={}
            for k,v in self._allParams.items():
                if "rfc" in k:
                    self._rfc_params[k]=v

                
            model = RandomForestClassifierPredictiveModel(self._X_train, 
                                                              self._y_train,
                                                              self._rfc_params,
                                                              self._nfolds, 
                                                              self._n_jobs,
                                                              self._verbose)
            return model.getBestEstimator().feature_importances_
        
        
        

        
        # Valid regressors
        regression_options = {"mixed_selection" : mixed_selection,
                              "pls":partialLeastSquaresRegression,
                               "svr" : supportVectorRegression,
                               "rfr":randomForestRegression,
                               "abr":adaptiveBoostingRegression,
                               "lasso":lassoRegression,
                               "enet":elasticNetRegression}



        # Valid classifiers
        classification_options = {'abc':adaptiveBoostingClassifier,
                                    'rfc':randomForestClassifier
                                    #,'svc':supportVectorClassifier
                                 }
        
		# Train test split
        self._X_train, self._X_test, self._y_train, self._y_test = train_test_split(self._X,
                                                                                     self._y,
                                                                                     test_size=self._test_size)

        
    
        # Wrapper models    
        self._key_features={}
        
        if self._regressors:
            for key in models_to_flow:
                 self._key_features[key]=regression_options[key]()
        elif self._regressors == False:
            for key in models_to_flow:
                 self._key_features[key]=classification_options[key]()
        else:
            print("Invalid model type. Please set regressors=True or regressors=False.")
            print
        if self._verbose:
            print
		
        if self._ensemble:
            print("ensemble == True")
        
        
        return self._key_features