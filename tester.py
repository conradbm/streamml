
import pandas as pd
import numpy as np
from streamline.transformation.flow.TransformationStream import TransformationStream
from streamline.model_selection.flow.ModelSelectionStream import ModelSelectionStream

# git checkout -b modelSelectionUpdates
# git push -u origin modelSelectionUpdates

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

print(X.shape)

Xnew = TransformationStream(X).flow(["scale","normalize","pca", "kmeans"], params={"pca__percent_variance":0.75, "kmeans__n_clusters":2})

#preproc options: scale, normalize, boxcox, binarize, pca, kmeans
#model options: lr, ridge, lasso, enet, svr, knnr, abr, rfr
#error options: 'mean_squared_error','r2'

#scoring option not working right, be okay with default scorers.
performances = ModelSelectionStream(Xnew,y).flow(["svr", "lr", "knnr","lasso","abr"],
                                              params={'svr__C':[1,0.1,0.01,0.001],
                                                      'svr__gamma':[0, 0.01, 0.001, 0.0001],
                                                      'svr__kernel':['poly', 'rbf'],
                                                     'lr__fit_intercept':[False, True],
                                                     'knnr__n_neighbors':[3, 5,7, 9, 11, 13],
                                                     'lasso__alpha':[0,0.01,1,10.0,20.0],
                                                     'abr__n_estimators':[10,20,50],
                                                     'abr__learning_rate':[0.1,1,10, 100]},
                                                metrics=['r2','rmse'],
                                                 verbose=True,
                                                regressors=True)
print(performances)
