from streamline.transformation.AbstractTransformer import *
from sklearn.manifold import TSNE

class TSNETransformer(AbstractTransformer):
    
    def __init__(self, ncomps):
        self._tsne_n_components = ncomps
        AbstractTransformer.__init__(self, "scale")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        X_embedded = TSNE(n_components=self._tsne_n_components).fit_transform(X)
        return pd.DataFrame(scale(X))