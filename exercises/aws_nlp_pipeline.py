"""
Start Date: 22 January 2019
Project: AWS NLP Pipeline
"""

from aws_nlp_loader import DataLoader
from aws_nlp_processer import * 
from aws_nlp_modeler import DataModeler
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.externals import joblib


class MachineLearningPipeline:
    """
    Class to 1) train model to produce risk score or 2) predict new risk score with new data
    """

    def __init__(self):
        # to load in basic values
        self.features = None
        self.response = None
        
        # store model
        self.model = None

    def load_data(self,
                 ):
        """
        Method to call DataLoader and load data into MachineLearningPipeline object
        """
        dl = DataLoader()
        dl.query_data()
        
        self.features = dl.features
        self.response = dl.response
    
    def gen_corpus(self):
        """
        Method to call DataCorpuser and process the data in the MachineLearningPipeline object
        """
        self.features = gen_bow_corpus(self.features)
        

    def model_data(self
                   ,in_mod_class=LatentDirichletAllocation()
                   ,persist_model=True):
        """
        Method to call DataModeler and create predictive model
        :param in_mod_class: sklearn model- model used for classification
        :return: saves out the data model
        """
        dm = DataModeler()
        dm.response = self.response
        dm.features = self.features
        dm.fit_model(mod_class=in_mod_class)
        self.model = dm.model
        if persist_model:
            dm.save_model()


def train_new_model():
    """
    Creates a class that loads and models data
    """
    #start = datetime.now()
    print('creating object')
    mlp = MachineLearningPipeline()
    print('loading data')
    mlp.load_data()
    print('generating bag of words corpus')
    mlp.gen_corpus()
    print('modeling data')
    mlp.model_data()
    print('done')
    #end = datetime.now()
    #print("Duration:", end - start)


if __name__ == '__main__':
    train_new_model()