
import pandas as pd
import numpy as np
from streamml.streamline.transformation.TransformationStream import TransformationStream

test = pd.DataFrame(np.matrix([[np.random.exponential() for j in range(10)] for i in range(20)]))

t = TransformationStream(test)
formed = t.flow(["scale","normalize","pca","binarize","kmeans"], params={"pca__percent_variance":0.75,
                                                                         "kmeans__n_clusters":3})
formed = t.flow(["scale","normalize", "pca"], params={"pca__percent_variance":0.75,
                                                      "kmeans__n_clusters":3})

formed = t.flow(["boxcox","scale","normalize"], verbose=True)
print formed
