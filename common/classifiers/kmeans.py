from sklearn import cluster
from common.familyclassifier import FamilyClassifier
import pandas as pd

from common.familynames import get_family_name

class KMeansFamilyClassifier(FamilyClassifier):
    """
    K-Means based asteroid family classifier.
    This is expected to do very poorly, since K-Means has no outlier rejection,
    and requires a given number of clusters, but it's a useful baseline.
    """
    def __init__(self, n_clusters: int = 39):
        """
        Initialize the KMeansFamilyClassifier with the given number of clusters.
        
        Parameters:
        n_clusters (int): The number of clusters to use for K-Means. Default is 39, which is the number of families in the training set.
        """
        self.n_clusters = int(n_clusters)
        self.kmeans = cluster.KMeans(n_clusters=self.n_clusters, random_state=42)
    
    @staticmethod
    def get_params() -> dict:
        """
        Get the parameters of the classifier as a dictionary. The keys should be the parameter names, and the values 
        should be a description of the parameter.
        
        Returns:
        dict: A dictionary containing the parameters of the classifier.
        """
        return {
            "n_clusters": "The number of clusters to use for K-Means. Default is 39, which is the number of families in the training set."
        }
    
    def classify(self, elements: pd.DataFrame) -> pd.DataFrame:
        """
        Classify the given asteroids and their proper elements into families using K-Means clustering.
        See FamilyClassifier.classify for detailed description of the parameters and return value.
        
        Parameters:
        elements (pd.DataFrame): A DataFrame containing the proper elements of asteroids. 
        
        Returns:
        pd.DataFrame: A DataFrame containing the family classification of the asteroids. 
            
        """
        # K-Means doesn't have a concept of "no family", so outliers will be assigned to the nearest cluster.
        results = self.kmeans.fit_predict(elements[["a", "e", "sinI", "n", "g", "s"]])
        # convert the cluster labels to family names. For each family, its name is the name of its earliest-sorted member.
        family_names = {}
        for i in range(self.n_clusters):
            cluster_members = elements[results == i]
            if len(cluster_members) == 0:
                family_names[i] = "0"
            else:
                family_names[i] = get_family_name(cluster_members["Name"])
        return pd.DataFrame({
            "name": elements["Name"],
            "family1": [family_names[i] for i in results]
        })