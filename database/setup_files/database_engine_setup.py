# Set up and import the actual data for our tables
"""
# Estimators the user can select from
class Estimator(Base):
    #F_Estimator_ID | PK
    #F_Estimator_Name | char(200)
    
    __tablename__ = 'T_Estimator'
    F_Estimator_ID = Column(Integer, primary_key=True)
    F_Estimator_Name = Column(String(250), nullable=False)
    F_Estimator_Symbol = Column(String(20), nullable=False)
    F_Estimator_PredictionClass = Column(String(20), nullable=False) # regressor or classifier
    F_Estimator_CanFeatureSelect = Column(Integer, nullable=False) # 1,0 if it can feature select

# All possible parameters
class Parameter(Base):
    
    __tablename__ = 'T_Parameter'
    F_Parameter_ID = Column(Integer, primary_key=True)
    F_Parameter_Open = Column(Integer, nullable=False)
    F_Parameter_Name = Column(String(20), nullable=False)
    F_Parameter_Description = Column(String(100), nullable=True)
    F_Estimator_ID = Column(Integer, ForeignKey('T_Estimator.F_Estimator_ID'))
    F_Estimator = relationship("Estimator", foreign_keys=['F_Estimator_ID'])
    
"""

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from database_setup import Base, Estimator, Parameter, ParameterValue

# Re-create the database
engine = create_engine('sqlite:///streamml.db')


# Relate Tables to DB
Base.metadata.bind = engine

# SQL Session Wrapper
DBSession = sessionmaker(bind=engine)
session = DBSession()


# Insert Regression Estimators
lr = Estimator(F_Estimator_Name = "Linear Regressor",
                   F_Estimator_Symbol = 'lr',
                   F_Estimator_PredictionClass = 'regressor',
               F_Estimator_CanFeatureSelect=0)


lr_param1 = Parameter(F_Estimator=lr,
                     F_Parameter_Open = 0,
                     F_Parameter_Name = 'fit_intercept',
                     F_Parameter_Description = 'whether to calculate the intercept for this model. If set to False, no intercept will be used in calculations (e.g. data is expected to be already centered)')

# Add lr
session.add(lr)
session.add(lr_param1)

svr = Estimator(F_Estimator_Name = "Support Vector Regressor",
                   F_Estimator_Symbol = 'svr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

svr_param1 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'C',
                      F_Parameter_Description = 'Penalty parameter C of the error term.'
                     )

svr_param2 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'epsilon',
                      F_Parameter_Description = 'Epsilon in the epsilon-SVR model. It specifies the epsilon-tube within which no penalty is associated in the training loss function with points predicted within a distance epsilon from the actual value.'
                     )
svr_param3 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'kernel',
                      F_Parameter_Description = 'Specifies the kernel type to be used in the algorithm. It must be one of ‘linear’, ‘poly’, ‘rbf’, ‘sigmoid’, ‘precomputed’ or a callable. If none is given, ‘rbf’ will be used. If a callable is given it is used to precompute the kernel matrix.'
                     )

svr_param4 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'degree',
                      F_Parameter_Description = 'Degree of the polynomial kernel function (‘poly’). Ignored by all other kernels.'
                     )
svr_param5 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'gamma',
                      F_Parameter_Description = 'Kernel coefficient for ‘rbf’, ‘poly’ and ‘sigmoid’. If gamma is ‘auto’ then 1/n_features will be used instead.'
                     )
svr_param6 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'coef0',
                      F_Parameter_Description = 'Independent term in kernel function. It is only significant in ‘poly’ and ‘sigmoid.'
                     )
svr_param7 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 0,
                      F_Parameter_Name = 'shrinking',
                      F_Parameter_Description = 'Whether to use the shrinking heuristic.'
                     )

svr_param8 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'tol',
                      F_Parameter_Description = 'Tolerance for stopping criterion.'
                     )

svr_param9 = Parameter(F_Estimator = svr,
                      F_Parameter_Open = 1,
                      F_Parameter_Name = 'max_iter',
                      F_Parameter_Description = 'Hard limit on iterations within solver, or -1 for no limit.'
                     )

# Add svr
session.add(svr)
session.add(svr_param1)
session.add(svr_param2)
session.add(svr_param3)
session.add(svr_param4)
session.add(svr_param5)
session.add(svr_param6)
session.add(svr_param7)
session.add(svr_param8)
session.add(svr_param9)


rfr = Estimator(F_Estimator_Name = "Random Forest Regressor",
                   F_Estimator_Symbol = 'rfr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

session.add(rfr)

abr = Estimator(F_Estimator_Name = "Adaptive Boosting Regressor",
                   F_Estimator_Symbol = 'abr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

session.add(abr)

knnr = Estimator(F_Estimator_Name = "K-Nearest Neighbors Regressor",
                   F_Estimator_Symbol = 'knnr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(knnr)

ridge = Estimator(F_Estimator_Name = "Ridge Regressor",
                   F_Estimator_Symbol = 'ridge',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(ridge)

lasso = Estimator(F_Estimator_Name = "Lasso Regressor",
                   F_Estimator_Symbol = 'lasso',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

session.add(lasso)
enet = Estimator(F_Estimator_Name = "ElasticNet Regressor",
                   F_Estimator_Symbol = 'enet',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

session.add(enet)

mlpr = Estimator(F_Estimator_Name = "Multi-Layer Perceptron Regressor",
                   F_Estimator_Symbol = 'mlpr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(mlpr)

br = Estimator(F_Estimator_Name = "Bagging Regressor",
                   F_Estimator_Symbol = 'br',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(br)

dtr = Estimator(F_Estimator_Name = "Decision Tree Regressor",
                   F_Estimator_Symbol = 'dtr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

session.add(dtr)

gbr = Estimator(F_Estimator_Name = "Gradient Boosting Regressor",
                   F_Estimator_Symbol = 'gbr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(gbr)

gpr = Estimator(F_Estimator_Name = "Gaussian Process Regressor",
                   F_Estimator_Symbol = 'gpr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(gpr)

hr = Estimator(F_Estimator_Name = "Huber Regressor",
                   F_Estimator_Symbol = 'hr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(hr)

tsr = Estimator(F_Estimator_Name = "Theil-Sen Regressor",
                   F_Estimator_Symbol = 'tsr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(tsr)

par = Estimator(F_Estimator_Name = "Passive Aggressive Regressor",
                   F_Estimator_Symbol = 'par',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(par)

ard = Estimator(F_Estimator_Name = "ARD Regressor",
                   F_Estimator_Symbol = 'ard',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(ard)

bays_ridge = Estimator(F_Estimator_Name = "Baysian Ridge Regressor",
                   F_Estimator_Symbol = 'bays_ridge',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0)

session.add(bays_ridge)

lasso_lar = Estimator(F_Estimator_Name = "Lasso Least Angle Regressor",
                   F_Estimator_Symbol = 'lasso_lar',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

session.add(lasso_lar)

lar = Estimator(F_Estimator_Name = "Least Angle Regressor",
                   F_Estimator_Symbol = 'lar',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1)

session.add(lar)

""" (Regressor Parameters)
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

"""

# Insert Classification Estimators
logr = Estimator(F_Estimator_Name = "Logistic Regression Classifier",
                   F_Estimator_Symbol = 'logr',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0)

session.add(logr)

svc = Estimator(F_Estimator_Name = "Support Vector Classifier",
                   F_Estimator_Symbol = 'svc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1)

session.add(svc)

rfc = Estimator(F_Estimator_Name = "Random Forest Classifier",
                   F_Estimator_Symbol = 'rfc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1)

session.add(rfc)

abc = Estimator(F_Estimator_Name = "Adaptive Boosting Classifier",
                   F_Estimator_Symbol = 'abc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1)

session.add(abc)

dtc = Estimator(F_Estimator_Name = "Decision Tree Classifier",
                   F_Estimator_Symbol = 'dtc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1)

session.add(dtc)

gbc = Estimator(F_Estimator_Name = "Gradient Boosting Classifier",
                   F_Estimator_Symbol = 'gbc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0)

session.add(gbc)

sgd = Estimator(F_Estimator_Name = "Stochastic Gradient Descent Classifier",
                   F_Estimator_Symbol = 'sgd',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0)

session.add(sgd)

gpc = Estimator(F_Estimator_Name = "Gaussian Process Classifier",
                   F_Estimator_Symbol = 'gpc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0)

session.add(gpc)

knnc = Estimator(F_Estimator_Name = "K-Nearest Neighbors Classifier",
                   F_Estimator_Symbol = 'knnc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0)

session.add(knnc)

mlpc = Estimator(F_Estimator_Name = "Multi-Layer Perceptron Classifier",
                   F_Estimator_Symbol = 'mlpc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0)

session.add(mlpc)

nbc = Estimator(F_Estimator_Name = "Naive Bayes Classifier",
                   F_Estimator_Symbol = 'nbc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0)

session.add(nbc)
""" (Classifier Parameters)

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html#sklearn.metrics.classification_report
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.cohen_kappa_score.html#sklearn.metrics.cohen_kappa_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.auc.html#sklearn.metrics.auc
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.log_loss.html#sklearn.metrics.log_loss
http://scikithttp://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html#sklearn.metrics.f1_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html#sklearn.metrics.precision_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html#sklearn.metrics.recall_score
http://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score

-learn.org/stable/modules/generated/sklearn.metrics.jaccard_similarity_score.html#sklearn.metrics.jaccard_similarity_score

http://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html#sklearn.metrics.confusion_matrix

"""

# ... Create Parameters for every estimator

# .. Add Parameters for every estimator


session.commit()
everything = session.query(Estimator).all()

for e in everything:
    print("%s\n\t%s\n\t%s\n\t%s\n\t%s" %(e.F_Estimator_ID,
                                           e.F_Estimator_Name, 
                                           e.F_Estimator_Symbol, 
                                           e.F_Estimator_PredictionClass, 
                                           e.F_Estimator_CanFeatureSelect) )

everything = session.query(Parameter).all()

print("\n")

for e in everything:
    print("%s\n\t%s\n\t%s\n\t%s\n\t%s" %(e.F_Parameter_ID, 
                                           e.F_Estimator_ID, 
                                           e.F_Parameter_Name, 
                                           e.F_Parameter_Open,
                                           e.F_Parameter_Description) )

# Query just the match ups
query = session.query(Estimator, Parameter).filter(Estimator.F_Estimator_ID == Parameter.F_Estimator_ID).all()
for e,p in query:
    print("%s (aka) %s\n\t%s" %(e.F_Estimator_Name,e.F_Estimator_Symbol, p.F_Parameter_Name))

# ... print to see if everything is working