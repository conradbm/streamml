#
#
#
#
# Model Selection Example
# test_model_selection_flow.py
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
X=pd.DataFrame(iris['data'], columns=iris['feature_names'])
y=pd.DataFrame(iris['target'], columns=['target'])



"""
Model Selection Params:
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
             modelSelection=False,
             cut=None):

Model Selection Models:
        # Valid regressors
        regression_options = {"lr" : linearRegression,
                               "svr" : supportVectorRegression,
                               "rfr":randomForestRegression,
                               "abr":adaptiveBoostingRegression,
                               "knnr":knnRegression,
                               "ridge":ridgeRegression,
                               "lasso":lassoRegression,
                               "enet":elasticNetRegression,
                               "mlpr":multilayerPerceptronRegression,
                               "br":baggingRegression,
                               "dtr":decisionTreeRegression,
                               "gbr":gradientBoostingRegression,
                               "gpr":gaussianProcessRegression,
                               "hr":huberRegression,
                               "tsr":theilSenRegression,
                               "par":passiveAggressiveRegression,
                               "ard":ardRegression,
                               "bays_ridge":bayesianRidgeRegression,
                               "lasso_lar":lassoLeastAngleRegression,
                               "lar":leastAngleRegression}



        # Valid classifiers
        classification_options = {'abc':adaptiveBoostingClassifier,
                                  'dtc':decisionTreeClassifier,
                                  'gbc':gradientBoostingClassifier,
                                    'gpc':guassianProcessClassifier,
                                    'knnc':knnClassifier,
                                    'logr':logisticRegressionClassifier,
                                    'mlpc':multilayerPerceptronClassifier,
                                    'nbc':naiveBayesClassifier,
                                    'rfc':randomForestClassifier,
                                    'sgd':stochasticGradientDescentClassifier,
                                    'svc':supportVectorClassifier}
"""

# Classification Test

"""
params={ 'abc__algorithm':['SAMME'],
                                                        'abc__base_estimator':[LogisticRegression(), SVC(), GaussianNB(), RandomForestClassifier()],
                                                        'abc__n_estimators':[50, 100, 150],
                                                        'rfc__n_estimators':[50, 100, 150],
                                                        'knnc__n_neighbors':[5,10,15],
                                                        'mlpc__hidden_layer_sizes':[(100), (100,100)],
                                                        'mlpc__alpha':[1e-5,1e-4,1e-3,1e-2,1e-1],
                                                        'mlpc__activation':['identity','logistic','relu','tanh'],
                      
                                  'mlpc__learning_rate':['constant','invscaling']}
["abc","rfc","logr","dtc", "gbc", "mlpc", "sgd","knnc"]
"""
classification_options = {'abc':0,
                          'dtc':0,
                          'gbc':0,
                            'gpc':0,
                            'knnc':0,
                            'logr':0,
                            'mlpc':0,
                            'nbc':0,
                            'rfc':0,
                            'sgd':0,
                            'svc':0}
best_models, scoring_results,final_results = ModelSelectionStream(X,y).flow(list(classification_options.keys()),
                                                                params={},
                                                                metrics=[],
                                                                test_size=0.35,
                                                                verbose=False, 
                                                                regressors=False,
                                                                stratified=True,
                                                                modelSelection=True,
                                                                n_jobs=3)

print("Best Models ... ")
print(best_models)
print("Metric Table ...")
print(pd.DataFrame(scoring_results))


#1. Does regression work?
#2. Why isn't n_jobs showing up for any of the classifiers?


from sklearn.datasets import load_boston
boston=load_boston()
X=pd.DataFrame(boston['data'], columns=boston['feature_names'])
y=pd.DataFrame(boston['target'],columns=["target"])
print(X.head())
print(y.head())
# Regression Test
regression_options={"lr" : 0,
                   "svr" : 0,
                   "rfr":0,
                   "abr":0,
                   "knnr":0,
                   "ridge":0,
                   "lasso":0,
                   "enet":0,
                   "mlpr":0,
                   "br":0,
                   "dtr":0,
                   "gbr":0,
                   "gpr":0,
                   "hr":0,
                   "tsr":0,
                   "par":0,
                   "ard":0,
                   "bays_ridge":0,
                   "lasso_lar":0,
                   "lar":0}
best_models, scoring_results, final_results = ModelSelectionStream(X,y).flow(list(regression_options.keys()),
                                                                    params={},
                                                                    metrics=[],
                                                                    verbose=True, 
                                                                    regressors=True,
                                                                    stratified=True, 
                                                                    cut=y['target'].mean(),
                                                                    modelSelection=True,
                                                                    n_jobs=3)

print("Best Models ... ")
print(best_models)
print("Metric Table ...")
print(pd.DataFrame(scoring_results))
print("Final Results ...")
print(pd.DataFrame(final_results))

print("Best Models ... ")
print(best_models)
print("Metric Table ...")
print(pd.DataFrame(scoring_results))
