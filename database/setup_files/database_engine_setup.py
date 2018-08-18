"""
********************************************************
*** Set up and import the actual data for our tables ***
********************************************************

************************
*** Table Structures ***
************************

class Estimator(Base):
    #F_Estimator_ID | PK
    #F_Estimator_Name | char(200)
    
    __tablename__ = 'T_Estimator'
    F_Estimator_ID = Column(Integer, primary_key=True)
    F_Estimator_Name = Column(String(250), nullable=False)
    F_Estimator_Symbol = Column(String(20), nullable=False)
    F_Estimator_PredictionClass = Column(String(20), nullable=False) # regressor or classifier
    F_Estimator_CanFeatureSelect = Column(Integer, nullable=False) # 1,0 if it can feature select

class Parameter(Base):
    
    __tablename__ = 'T_Parameter'
    F_Parameter_ID = Column(Integer, primary_key=True)
    F_Estimator_ID = Column(Integer, ForeignKey('T_Estimator.F_Estimator_ID'))
    F_Parameter_Open = Column(Integer, nullable=False)
    F_Parameter_Name = Column(String(20), nullable=False)
    F_Parameter_Description = Column(String(100), nullable=True)

    #Relationship From
    F_Estimator = relationship(Estimator)


class ParameterValue(Base):
    #F_ParameterValue_ID | PK
    #F_ParameterValue_Realization | Char(20); Actual value the user selected for the parameter
    
    __tablename__ = 'T_ParameterValue'
    F_ParameterValue_ID = Column(Integer, primary_key=True)
    F_Parameter_ID = Column(Integer , ForeignKey('T_Parameter.F_Parameter_ID'))
    F_ParameterValue_Realization = Column(String(10), nullable=False)

    
     #Relationship From
    F_Parameter = relationship(Parameter)
    
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




models=[]
# Insert Regression Estimators
lr = Estimator(F_Estimator_Name = "Linear Regressor",
                   F_Estimator_Symbol = 'lr',
                   F_Estimator_PredictionClass = 'regressor',
               F_Estimator_CanFeatureSelect=0,
              F_Estimator_Description="Ordinary least squares Linear Regression.")


# Add lr
models.append(lr)

svr = Estimator(F_Estimator_Name = "Support Vector Regressor",
                   F_Estimator_Symbol = 'svr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description="Epsilon-Support Vector Regression. The free parameters in the model are C and epsilon. The implementation is based on libsvm.")

# Add svr
models.append(svr)

rfr = Estimator(F_Estimator_Name = "Random Forest Regressor",
                   F_Estimator_Symbol = 'rfr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                F_Estimator_Description = "A random forest is a meta estimator that fits a number of classifying decision trees on various sub-samples of the dataset and use averaging to improve the predictive accuracy and control over-fitting. The sub-sample size is always the same as the original input sample size but the samples are drawn with replacement if bootstrap=True (default).")


# Add rfr
models.append(rfr)

abr = Estimator(F_Estimator_Name = "Adaptive Boosting Regressor",
                   F_Estimator_Symbol = 'abr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "An AdaBoost regressor is a meta-estimator that begins by fitting a regressor on the original dataset and then fits additional copies of the regressor on the same dataset but where the weights of instances are adjusted according to the error of the current prediction. As such, subsequent regressors focus more on difficult cases. This class implements the algorithm known as AdaBoost.R2.")

# Add abr
models.append(abr)

knnr = Estimator(F_Estimator_Name = "K-Nearest Neighbors Regressor",
                   F_Estimator_Symbol = 'knnr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")

# add knnr
models.append(knnr)

ridge = Estimator(F_Estimator_Name = "Ridge Regressor",
                   F_Estimator_Symbol = 'ridge',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                 F_Estimator_Description = "")

# add ridge
models.append(ridge)


lasso = Estimator(F_Estimator_Name = "Lasso Regressor",
                   F_Estimator_Symbol = 'lasso',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                 F_Estimator_Description = "Linear Model trained with L1 prior as regularizer (aka the Lasso). The optimization objective for Lasso is: (1 / (2 * n_samples)) * ||y - Xw||^2_2 + alpha * ||w||_1")

"""
lasso_param1 = Parameter(F_Estimator = lasso,
                       F_Parameter_Open = 1,
                       F_Parameter_Name = 'alpha',
                       F_Parameter_Description='float, Constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is equivalent to an ordinary least square, solved by the LinearRegression object. For numerical reasons, using alpha = 0 with the Lasso object is not advised. Given this, you should use the LinearRegression object.')
"""                   

# add lasso
models.append(lasso)

enet = Estimator(F_Estimator_Name = "ElasticNet Regressor",
                   F_Estimator_Symbol = 'enet',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                F_Estimator_Description = "")

models.append(enet)

mlpr = Estimator(F_Estimator_Name = "Multi-Layer Perceptron Regressor",
                   F_Estimator_Symbol = 'mlpr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")

models.append(mlpr)

br = Estimator(F_Estimator_Name = "Bagging Regressor",
                   F_Estimator_Symbol = 'br',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
              F_Estimator_Description = "")

models.append(br)

dtr = Estimator(F_Estimator_Name = "Decision Tree Regressor",
                   F_Estimator_Symbol = 'dtr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(dtr)

gbr = Estimator(F_Estimator_Name = "Gradient Boosting Regressor",
                   F_Estimator_Symbol = 'gbr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gbr)

gpr = Estimator(F_Estimator_Name = "Gaussian Process Regressor",
                   F_Estimator_Symbol = 'gpr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gpr)


hr = Estimator(F_Estimator_Name = "Huber Regressor",
                   F_Estimator_Symbol = 'hr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
              F_Estimator_Description = "")


models.append(hr)

tsr = Estimator(F_Estimator_Name = "Theil-Sen Regressor",
                   F_Estimator_Symbol = 'tsr',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(tsr)

par = Estimator(F_Estimator_Name = "Passive Aggressive Regressor",
                   F_Estimator_Symbol = 'par',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(par)

ard = Estimator(F_Estimator_Name = "ARD Regressor",
                   F_Estimator_Symbol = 'ard',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(ard)

bays_ridge = Estimator(F_Estimator_Name = "Baysian Ridge Regressor",
                   F_Estimator_Symbol = 'bays_ridge',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 0,
                      F_Estimator_Description = "")


models.append(bays_ridge)

lasso_lar = Estimator(F_Estimator_Name = "Lasso Least Angle Regressor",
                   F_Estimator_Symbol = 'lasso_lar',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
                     F_Estimator_Description = "")


models.append(lasso_lar)

lar = Estimator(F_Estimator_Name = "Least Angle Regressor",
                   F_Estimator_Symbol = 'lar',
                   F_Estimator_PredictionClass = 'regressor',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(lar)

# Insert Classification Estimators
logr = Estimator(F_Estimator_Name = "Logistic Regression Classifier",
                   F_Estimator_Symbol = 'logr',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")


models.append(logr)

svc = Estimator(F_Estimator_Name = "Support Vector Classifier",
                   F_Estimator_Symbol = 'svc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(svc)

rfc = Estimator(F_Estimator_Name = "Random Forest Classifier",
                   F_Estimator_Symbol = 'rfc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(rfc)

abc = Estimator(F_Estimator_Name = "Adaptive Boosting Classifier",
                   F_Estimator_Symbol = 'abc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(abc)

dtc = Estimator(F_Estimator_Name = "Decision Tree Classifier",
                   F_Estimator_Symbol = 'dtc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 1,
               F_Estimator_Description = "")


models.append(dtc)

gbc = Estimator(F_Estimator_Name = "Gradient Boosting Classifier",
                   F_Estimator_Symbol = 'gbc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gbc)

sgd = Estimator(F_Estimator_Name = "Stochastic Gradient Descent Classifier",
                   F_Estimator_Symbol = 'sgd',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(sgd)

gpc = Estimator(F_Estimator_Name = "Gaussian Process Classifier",
                   F_Estimator_Symbol = 'gpc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "")


models.append(gpc)

knnc = Estimator(F_Estimator_Name = "K-Nearest Neighbors Classifier",
                   F_Estimator_Symbol = 'knnc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")


models.append(knnc)

mlpc = Estimator(F_Estimator_Name = "Multi-Layer Perceptron Classifier",
                   F_Estimator_Symbol = 'mlpc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
                F_Estimator_Description = "")


models.append(mlpc)

nbc = Estimator(F_Estimator_Name = "Naive Bayes Classifier",
                   F_Estimator_Symbol = 'nbc',
                   F_Estimator_PredictionClass = 'clasifier',
                   F_Estimator_CanFeatureSelect = 0,
               F_Estimator_Description = "Gaussian Naive Bayes (GaussianNB). Can perform online updates to model parameters via partial_fit method. For details on algorithm used to update feature means and variance online, see Stanford CS tech report STAN-CS-79-773 by Chan, Golub, and LeVeque: http://i.stanford.edu/pub/cstr/reports/cs/tr/79/773/CS-TR-79-773.pdf")


models.append(nbc)

links = ["http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVR.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html",
         "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lasso.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNet.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.HuberRegressor.html#sklearn.linear_model.HuberRegressor",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.TheilSenRegressor.html#sklearn.linear_model.TheilSenRegressor",
                 "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.PassiveAggressiveRegressor.html#sklearn.linear_model.PassiveAggressiveRegressor",
        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ARDRegression.html#sklearn.linear_model.ARDRegression",

        "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.BayesianRidge.html#sklearn.linear_model.BayesianRidge",
"http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LassoLars.html",
         "http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Lars.html",
         
"http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.SGDClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html",
"http://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html",
         "http://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.GaussianNB.html"

]

import requests
import re
import string
printable = set(string.printable)
regex_estimators1 = r'<dd><p>(.*)<\/p>'
regex_estimators2 = r'<p>(.*)<\/p>'
regex_estimators_flag = False
regex_parameters = r'<p.*><strong>(.*)<\/strong> : (.*)<\/p>'

# Make sure the links line up with the models
#for i,j in zip(models, links):
#    print(i.F_Estimator_Name,j)
#input("...")

link_contents = []
for i,link in enumerate(links):
    link_contents.append(str(requests.get(link).content).split("\\n"))
    #print(link_contents[-1])
    #print(link_contents[-1].replace("\\n","")[:100])
    #print("****%s****" %(link[-30:]))
    
    # Find the description of the model, its the first <dd> tag folowed by the next <p>, so it looks funny.
    estimator_description = ""
    for thing in link_contents[-1]:
        #print(thing)
        if regex_estimators_flag == False:
            results_est = re.findall(regex_estimators1,thing)
        else:
            results_est = re.findall(regex_estimators2,thing)
        
        if len(results_est) > 0:
            
            if regex_estimators_flag == False:
                hits = list(map(lambda x: x, results_est))[0]
                
                estimator_description += hits
                #print("a")
                #input(estimator_description)
                regex_estimators_flag = True
                break
            else:
                
                # Process
                hits = list(map(lambda x: x, results_est))[0]
                estimator_description += hits
                regex_estimators_flag=False
                #print("b")
                #input(estimator_description)
                break
            

    # Update description of models by specific regex foolishness
    estimator_description = param_descr = re.sub(r'\\x\d*|e2|<.*>|<|>', '', estimator_description)
    models[i].F_Estimator_Description=estimator_description
    session.add(models[i])
    
    for thing in link_contents[-1]:
        
        results_para = re.findall(regex_parameters, thing)
        if len(results_para) > 0:
            #print(thing)
            #print(list(map(lambda x: (x[0],x[1]), results)))
            #input("...")
            hits = list(map(lambda x: (x[0],x[1]), results_para))[0]
            param_name = hits[0]
            param_descr = re.sub(r'\\x\d*|e2|<.*>|<|>', '', hits[1])
            
            if any([param_name in i for i in ["n_jobs", "random_state", "verbose", "copy_X", "copy_X_train", "cache_size"]]):
                continue
            if 'Attributes:' in thing:
                break
            #if param_name == "X":
            #    break
            param_open = 1
            if any([i in param_descr for i in ['str', 'string', 'bool', 'boolean']]):
                param_open = 0
                
            param = Parameter(F_Estimator = models[i],
                       F_Parameter_Open = param_open,
                       F_Parameter_Name = param_name,
                       F_Parameter_Description=param_descr)
            session.add(param)
            
    #input("...")



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
    print("%s (aka) %s\n\t%s(%s)\t%s" %(e.F_Estimator_Name,e.F_Estimator_Symbol, p.F_Parameter_Name, p.F_Parameter_Open, p.F_Parameter_Description))


