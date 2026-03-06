"""
An abstract base class for classifiers to derive from.
"""
from abc import ABC, abstractmethod
import pandas as pd

class FamilyClassifier(ABC):
    @abstractmethod
    def classify(self, elements: pd.DataFrame) -> pd.DataFrame:
        """
        Classify the given asteroids and their proper elements into families.
        
        Parameters:
        elements (pd.DataFrame): A DataFrame containing the proper elements of asteroids. 
            Columns: 'Name', 'a', 'e', 'sinI', 'n', 'g', 's'. 'Name' is a string, the others are floats.
        Returns:
        pd.DataFrame: A DataFrame containing the family classification of the asteroids. 
            Columns: 'name', 'family1', both strings. Family1 is "0" if the asteroid is not a member of any family,
            or the name of the family it is a member of. (The name of a family is the name of its lowest numbered member.)
        """
        pass
    @abstractmethod
    def get_params() -> dict:
        """
        Static method.
        Get the parameters of the classifier as a dictionary. The keys should be the parameter names, and the values 
        should be a description of the parameter.
        """
        pass