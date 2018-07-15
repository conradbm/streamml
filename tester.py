
import pandas as pd
import numpy as np

# FOR ANY SYSTEM: INCLUDE STREAMML
import sys
sys.path.append('/Users/bmc/Desktop/') #I.e., make it a path variable

from streamml.streamline.transformation.flow.TransformationStream import TransformationStream
from streamml.streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream

# git checkout -b modelSelectionUpdates
# git push -u origin modelSelectionUpdates

# FOR MAC:
# nano ~/.bash_profile
# export PYTHONPATH="${PYTHONPATH}:/Users/bmc/Desktop/streamml"
# source ~/.bash_profile
# python -W ignore tester.py

X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))
y = pd.DataFrame(np.array([np.random.exponential() for i in range(200)]))

#D = pd.read_csv("Series3_6.15.17_padel.csv")
#X = D.iloc[:,2:]
#y = D.iloc[:,1]

#ynakiller = y.isna()
#X = X.loc[-ynakiller,:]
#y = y.loc[-ynakiller]
#X.replace([np.nan, np.inf, -np.inf],0, inplace=True)

#print(X.shape)
#print (y.shape)

"""
Transformation Options:
["scale","normalize","boxcox","binarize","pca","kmeans", "brbm]
kmeans: n_clusters
pca: percent_variance (only keeps # comps that capture this %)
binarize: threshold (binarizes those less than threshold as 0 and above as 1)
tsne: n_components

# sklearn.decomposition.sparse_encode
# sklearn.preprocessing.PolynomialFeatures
# sklearn.linear_model.OrthogonalMatchingPursuit

"""
Xnew = TransformationStream(X).flow(["scale","tsne"], 
                                    params={"tnse_n_components":4,
                                            "pca__percent_variance":0.75, 
                                            "kmeans__n_clusters":2},
                                   verbose=True)
print(Xnew)




"""
Model Selection Options:
        options = {"lr" : linearRegression,
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

Model parameter options can all be found here in the following links for the model you wish to flow on:
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html
http://scikit-learn.org/stable/modules/gaussian_process.html#gaussian-process-regression-gpr
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html#sklearn.linear_model.LassoLars
http://scikit-learn.org/stable/modules/linear_model.html#least-angle-regression

Metric Options:
['rmse','mse', 'r2','explained_variance','mean_absolute_error','median_absolute_error']
"""

# Complex Example 

performances = ModelSelectionStream(Xnew,y).flow(["svr", "lr", "knnr","lasso","abr", "ridge","enet", "rfr", "mlpr"],
                                              params={'svr__C':[1,0.1,0.01,0.001],
                                                      'svr__gamma':[0, 0.01, 0.001, 0.0001],
                                                      'svr__kernel':['poly', 'rbf'],
                                                      'svr__epsilon':[0,0.1,0.01,0.001],
                                                      'svr__degree':[1,2,3,4,5,6,7],
                                                      'lr__fit_intercept':[False, True],
                                                      'knnr__n_neighbors':[3, 5,7, 9, 11, 13],
                                                      'lasso__alpha':[0, 0.1, 0.01,1,10.0,20.0],
                                                      'ridge__alpha':[0, 0.1, 0.01,1,10.0,20.0],
                                                      'enet__alpha':[0, 0.1, 0.01,1,10,20],
                                                      'enet__l1_ratio':[.25,.5,.75],
                                                      'abr__n_estimators':[10,20,50],
                                                      'abr__learning_rate':[0.1,1,10, 100],
                                                      'rfr__criterion':['mse', 'mae'],
                                                      'rfr__n_estimators':[10,100,1000],
                                                      'mlpr__hidden_layer_sizes':[(Xnew.shape[1], Xnew.shape[1]/2, Xnew.shape[1]/4),
                                                                                  (100,10,2),
                                                                                  (1000,100,10,1)]},
                                                 metrics=['r2','rmse', 'mse',
                                                          'explained_variance','mean_absolute_error',
                                                         'median_absolute_error'],
                                                verbose=True,
                                                regressors=True,
                                                cut=2) # evenly splits folds with points beneath cut.


performances
