from streamline.transformation.AbstractTransformer import *
from sklearn.preprocessing import normalize

class NormalizeTransformer(AbstractTransformer):
    
    def __init__(self):
        AbstractTransformer.__init__(self, "scale")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
        return pd.DataFrame(normalize(X, norm='l2'))