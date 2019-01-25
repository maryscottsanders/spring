
"""
Start Date: 22 January 2019
Project AWS NLP Pipeline
"""

from sklearn.datasets import fetch_20newsgroups

class DataLoader:

    def __init__(self):
        # to load in basic values
        self.features = None
        self.response = None

    def query_data(self):
        """
        Method to populate object with data in a structured csv file
        :return: updates self with features
        """
        data = fetch_20newsgroups(subset='train', shuffle = True)
        self.features = data.data
        self.response = data.target
        print('loaded')