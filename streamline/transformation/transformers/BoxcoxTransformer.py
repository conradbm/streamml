    
from streamline.transformation.AbstractTransformer import *
from scipy import stats

class BoxcoxTransformer(AbstractTransformer):
    
    def __init__(self):
        AbstractTransformer.__init__(self, "scale")
        
    # More parameters can be found here: 
    # http://scikit-learn.org/stable/modules/preprocessing.html
    def transform(self, X):
         X_boxcoxed = X
            lambdas=[]
            for col in X:
                X_boxcoxed[col], l = stats.boxcox(X_boxcoxed[col])
                lambdas.append(l)
                
            self._lambdas = lambdas
            
            if verbose:
                print("Optimized BoxCox-Lambdas For Each Column: ")
                print(self._lambdas)

        return pd.DataFrame(X_boxcoxed)
        