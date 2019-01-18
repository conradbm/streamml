#
#
#
#
# Transformation Flow Example
# test_transformation_flow.py
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

from streamml.streamline.transformation.flow.TransformationStream import TransformationStream
from sklearn.datasets import load_iris
iris=load_iris()
X=pd.DataFrame(iris['data'], columns=iris['feature_names'])
y=pd.DataFrame(iris['target'], columns=['target'])
"""
Transformation Selection Params:
    def flow(self, preproc_args=[], params=None, verbose=False):

Transformation Selection Options:
        # map the inputs to the function blocks
        options = {"scale" : runScale,
                   "normalize" : runNormalize,
                   "binarize" :runBinarize,
                   "itemset": runItemset,
                   "boxcox" : runBoxcox,
                   "pca" : runPCA,
                   "kmeans" : runKmeans,
                  "brbm": runBRBM,
                  "tsne":runTSNE}

kmeans, pca, and tsne have standard sklearn parameters available.
pca has the additional  pca__percent_variance option which keeps only the components representing the specified percentage of variance given.
"""

X2 = TransformationStream(X).flow(["pca","normalize","kmeans"],
                                  params={"pca__percent_variance":0.95,
                                          "kmeans__n_clusters":len(set(y[0]))}, 
                                  verbose=True)
print(X2)