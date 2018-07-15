from streamline.transformation.AbstractTransformer import *
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class PCATransformer(AbstractTransformer):
    
    def __init__(self, percent_variance, verbose=False):
        self._percent_variance = percent_variance
        self._verbose = verbose
        AbstractTransformer.__init__(self, "scale")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        pca = PCA()
            
        pca_output = pca.fit(X)
        components_nums = [i for i in range(len(pca_output.components_))]
        self._percentVarianceExplained = pca_output.explained_variance_/sum(pca_output.explained_variance_)
            
        tot=0.0
        idx=0
        for i in range(len(self._percentVarianceExplained)):
            tot+= self._percentVarianceExplained[i]
            if tot >= self._percent_variance:
                idx=i+1
                break
        if self._verbose:
            print ("Percent of Variance Explained By Components:\n")
            print (str(self._percentVarianceExplained), "\n")
            print (str(self._percent_variance*100), "% variance is explained in the first ", str(idx), " components\n")
            pca_df=pd.DataFrame({'explained_variance':pca_output.explained_variance_}, index=components_nums)
            pca_df.plot(title='Components vs. Variance')
            plt.show()
            
        pca_output = pca.fit_transform(X)
        pca_df = pd.DataFrame(pca_output, columns=["PC_"+str(i) for i in components_nums])
            
        return pca_df.iloc[:, :idx]