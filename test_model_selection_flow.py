#
#
#
#
# Feature Selection Example
# test_feature_selection_flow.py
#
#
#
#
#
#

import pandas as pd
import numpy as np
import os
import sys
sys.path.append(os.getcwd()) #I.e., make it a path variable
sys.path.append(os.path.join(os.getcwd(),"streamml"))

from streamml.streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_iris

iris=load_iris()
X=pd.DataFrame(iris['data'])
y=pd.DataFrame(iris['target'])


"""
1. Does regression work?
2. Why isn't n_jobs showing up for any of the classifiers?
"""
from sklearn.datasets import load_boston
boston=load_boston()
X=pd.DataFrame(boston['data'])
y=pd.DataFrame(boston['target'])

"""
Feature Selection Params:
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
             cut=None,
             ensemble=False):

Feature Selection Models:
        #Valid regressors
        regression_options = {"mixed_selection" : mixed_selection,
                               "svr" : supportVectorRegression,
                               "rfr":randomForestRegression,
                               "abr":adaptiveBoostingRegression,
                               "lasso":lassoRegression,
                               "enet":elasticNetRegression,
                               "plsr":partialLeastSquaresRegression}
        # Valid classifiers
        classification_options = {'abc':adaptiveBoostingClassifier,
                                    'rfc':randomForestClassifier,
                                    'svc':supportVectorClassifier
                                 }
"""


best_models, scoring_results = ModelSelectionStream(X,y).flow(["abc","rfc","logr","dtc", "gbc", "mlpc", "sgd", "knnc"],
                                    					 params={ 'abc__algorithm':['SAMME'],
                                                        'abc__base_estimator':[LogisticRegression(), SVC(), GaussianNB(), RandomForestClassifier()],
                                                        'abc__n_estimators':[50, 100, 150],
                                                        'rfc__n_estimators':[50, 100, 150],
                                                        'knnc__n_neighbors':[5,10,15],
                                                        'mlpc__hidden_layer_sizes':[(100), (100,100)],
                                                        'mlpc__alpha':[1e-5,1e-4,1e-3,1e-2,1e-1],
                                                        'mlpc__activation':['identity','logistic','relu','tanh'],
                                                        'mlpc__learning_rate':['constant','invscaling']},
                                    					 metrics=["precision", "accuracy", "recall", "f1", "kappa"],
                                               verbose=False, 
                                               regressors=False,
                                               modelSelection=True,
                                               n_jobs=3)
print("Best Models ... ")
print(best_models)
print("Metric Table ...")
print(pd.DataFrame(scoring_results))