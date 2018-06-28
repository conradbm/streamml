<h1>Streamlined Machine Learning</h1>
<hr>
<strong>Streamlined Machine Learning</strong> is a set of robust functions and classes meant to streamline: preprocessing, model selection, and feature selection.
This package is build on top of <em>SciPy</em> and <em>sklearn</em>.

<h2>Basic Usage</h2>
By building a <code>Stream</code> object, you can specify a list of predefined objects the package manages, then you can <code>flow</code> through them each on default grid selection parameters or user defined parameters (denoted <code>params</code>).
Streams provided:
<ul>

  <li><code>TransformationStream</code>, meant to flow through preprocessing techniques such as: scaling, normalizing, boxcox, binarization, pca, or kmeans aimed at returning a desired input dataset for model development.</li>

  <li><code>ModelSelectionStream</code>, meant to flow through several predictive models to determine which is the best, these include: LinearRegression, SupportVectorRegressor, RandomForestRegressor, KNNRegressor, AdaptiveBoostingRegressor, LassoRegressor, RidgeRegressor, and ElasticNetRegressor. You must specify whether your steam is a <em>regressor</em> or <em>classifier</em> stream (denoted <code>regressor=True</code>. Error metrics currently supported are Root Mean Squared Error ('rmse'), Mean Squared Error ('mse'), R2 ('r2'), Explained Variance, ('explained_variance'),Mean Absolute Error ('mean_absolute_error'), and Median Absolute Error ('median_absolute_error'])</li>

  <li><code>FeatureSelectionStream</code>, meant to flow through several predictive models and algorithms to determine which subset of features is most predictive or representative of your dataset, these include: RandomForestFeatureImportance, LassoFeatureImportance, MixedSelection, and a technique to ensemble each named TOPSISFeatureRanking. You must specify whether your wish to ensemble and with what technique (denoted <code>ensemble=True)</code> 
  </li>
</ul>

<hr>

<h2>Current Implementation</h2>

Currently we support transformation streams and restricted model selection streams with 8 regression estiminators.

Example of a transformation stream:

<strong>Simple data set</strong>

<code>
X = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(200)]))

y = pd.DataFrame(np.array([np.random.exponential() for i in range(200)]))
</code>


<strong>Supported stream operators</strong>: scale, normalize, boxcox, binarize, pca, kmeans, brbm (Bernoulli Restricted Boltzman Machine).


<code> 
Xnew = TransformationStream(X).flow(
                                    #required: list of transformations to pipe through
                                    ["scale","normalize","pca", "binarize", "boxcox", "kmeans", "brbm"], 
                                    
                                    #optional: parameters to specific transformers, NOT A GRIDSEARCH
                                    params={"pca__percent_variance":0.75, 
                                            "kmeans__n_clusters":2, 
                                            "binarize__threshold":0.5, 
                                            "brbm__n_components":X.shape[1], 
                                            "brbm__learning_rate":0.0001},
                                            
                                   #optional: displays graphics
                                   verbose=True)
                                   
</code>


  
<strong>Supported stream operators</strong>: lr, ridge, lasso, enet, svr, knnr, abr, mlpr

<strong>Supported Args</strong>: metrics -> ['rmse','mse', 'r2','explained_variance','mean_absolute_error','median_absolute_error'], verbose, regressors, and params. Params is build into the GridSearchCV function within sklearn, so each specified parameter will be automatically plugged into this method and hypertuned for you.

<code>
performances = ModelSelectionStream(Xnew,y).flow(
                                              
                                              #required: models you want to flow through in model selection
                                              ["svr", "lr", "knnr","lasso","abr","mlp","enet"],
                                              
                                              #optional: sklearn `.fit` enabled model object parameters
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
                                                        
                                                 'rfr__n_estimators':[10,100,1000]}, 
                                                 
                                                 
                                                 #optional: metrics you want competing models to draw against, if none specified no plot
                                                 metrics=['r2','rmse', 'mse',
                                                          'explained_variance','mean_absolute_error',
                                                         'median_absolute_error'],
                                                
                                                #optional: displays graphics
                                                verbose=True,
                                                
                                                #required: true if you want a regressor-like response, false if classification label based response
                                                regressors=True,
                                                
                                                #required: only required for competing regressors, equally distributes points less than cut into stratified k-folds during competition.
                                                cut=2)
                                                 
</code>


