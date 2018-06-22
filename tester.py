
import pandas as pd
import numpy as np

# FOR ANY SYSTEM: INCLUDE STREAMML
#import sys
#sys.path.insert(2, 'C:\\Users\\1517766115.CIV\\Desktop\\streamml')
from streamline.transformation.flow.TransformationStream import TransformationStream
from streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream

# git checkout -b modelSelectionUpdates
# git push -u origin modelSelectionUpdates

# FOR MAC:
# nano ~/.bash_profile
# export PYTHONPATH="${PYTHONPATH}:/Users/bmc/Desktop/streamml"
# source ~/.bash_profile
# python -W ignore tester.py

#X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))
#y = pd.DataFrame(np.array([np.random.exponential() for i in range(200)]))

D = pd.read_csv("Series3_6.15.17_padel.csv")
X = D.iloc[:,2:]
y = D.iloc[:,1]

ynakiller = y.isna()
X = X.loc[-ynakiller,:]
y = y.loc[-ynakiller]
X.replace([np.nan, np.inf, -np.inf],0, inplace=True)

#print(X.shape)
#print (y.shape)

"""
Supported Transformations:
["scale","normalize","boxcox","binarize","pca","kmeans", "brbm]
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Normalizer.html
https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.boxcox.html
http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.Binarizer.html
http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.PCA.html
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.BernoulliRBM.html#sklearn.neural_network.BernoulliRBM

["brbm"] --> Latent representations of the data

"""
Xnew = TransformationStream(X).flow(["scale", "normalize", "pca"], 
                                    params={"pca__percent_variance":0.75, 
                                            "kmeans__n_clusters":2,
                                            "binarize__threshold":0.2,
                                           "brbm__learning_rate":0.001,
                                           "brbm__n_components":X.shape[0]},
                                   verbose=True)

#preproc options: scale, normalize, boxcox, binarize, pca, kmeans
#model options: 
#error options: 'mean_squared_error','r2'

#scoring option not working right, be okay with default scorers.

"""
Supported Models:
["lr", "ridge", "lasso", "enet", "svr", "knnr", "abr", "rfr"]
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html
http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html
http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html
http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html
http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html

To be implemented
["mlpr", "dtr"]
http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html#sklearn.neural_network.MLPRegressor
http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html#sklearn.tree.DecisionTreeRegressor

Supported Metrics:
['rmse','mse', 'r2','explained_variance','mean_absolute_error','median_absolute_error']
"""

performances = ModelSelectionStream(Xnew,y).flow(["svr", "lr", "knnr","lasso","abr", "ridge","enet", "rfr"],
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
                                                 'rfr__n_estimators':[10,100,1000]}, # any any other sklearn parameter you want!
                                                 metrics=['r2','rmse', 'mse',
                                                          'explained_variance','mean_absolute_error',
                                                         'median_absolute_error'],
                                                verbose=True,
                                                regressors=True,
                                                cut=2) # cut is only required for regressors
                                                
print(performances)
