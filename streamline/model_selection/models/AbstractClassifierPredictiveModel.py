
from streamml.streamline.model_selection.AbstractPredictiveModel import AbstractPredictiveModel

class AbstractClassifierPredictiveModel(AbstractPredictiveModel):

    #constructor
    def __init__(self):
        print ("Constructed AbstractClassifierPredictiveModel")
        super("classifier")
    #methods
    def execute(self):
        pass