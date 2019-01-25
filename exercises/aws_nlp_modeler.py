"""
Start Date: 22 January 2019
Project: AWS NLP Pipeline
"""

#import pandas as pd
from sklearn.externals import joblib

class DataModeler:

    def __init__(self):
        # to load in basic values
        self.response = None
        self.features = None

        # store model
        self.model = None

    def fit_model(self, mod_class):
        """
        Method to train a new model
        :param mod_class: sklearn model- model to use for classification
        :return: model object is saved to file
        """
        self.model = mod_class.fit(self.features)

    def save_model(self):
        """
        Method to save current model iteration
        :return: saves model locally
        """
        joblib.dump(self.model, 'dev_model.pkl')
