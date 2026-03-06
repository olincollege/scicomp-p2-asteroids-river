from sklearn import cluster
from common.familyclassifier import FamilyClassifier
import pandas as pd
from common.familynames import get_family_name

class DBSCANFamilyClassifier(FamilyClassifier):
    """
    DBSCAN based asteroid family classifier.
    DBSCAN is a density-based clustering algorithm that can find clusters of arbitrary shape and can identify outliers.
    """
    def __init__(self, eps: float = 0.01, min_samples: int = 5):
        """
        Initialize the DBSCANFamilyClassifier with the given parameters.
        
        Parameters:
        eps (float): The maximum distance between two samples for them to be considered as in the same neighborhood. Default is 0.01, but this absolutely requires tuning.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Default is 5, which is a common choice for DBSCAN.
        """
        self.eps = float(eps)
        self.min_samples = int(min_samples)
        self.dbscan = cluster.DBSCAN(eps=self.eps, min_samples=self.min_samples)
    
    @staticmethod
    def get_params() -> dict:
        """
        Get the parameters of the classifier as a dictionary. The keys should be the parameter names, and the values 
        should be a description of the parameter.
        
        Returns:
        dict: A dictionary containing the parameters of the classifier.
        """
        return {
            "eps": "The maximum distance between two samples for them to be considered as in the same neighborhood. Default is 0.01, but this will requires tuning.",
            "min_samples": "The number of samples in a neighborhood for a point to be considered as a core point. Default is 5, which is a common choice for DBSCAN."
        }
    
    def classify(self, elements: pd.DataFrame) -> pd.DataFrame:
        """
        Classify the given asteroids and their proper elements into families using DBSCAN clustering.
        See FamilyClassifier.classify for detailed description of the parameters and return value.
        
        Parameters:
        elements (pd.DataFrame): A DataFrame containing the proper elements of asteroids. 
        
        Returns:
        pd.DataFrame: A DataFrame containing the family classification of the asteroids. 
            
        """
        results = self.dbscan.fit_predict(elements[["a", "e", "sinI", "n", "g", "s"]])
        print(f"DBSCAN found {len(set(results)) - (1 if -1 in results else 0)} clusters and {list(results).count(-1)} outliers.")
        # convert the cluster labels to family names. For each family, its name is the name of its earliest-sorted member. The outliers (results == -1) are assigned to family "0".
        family_names = {}
        for i in set(results):
            if i == -1:
                family_names[i] = "0"
            else:
                cluster_members = elements[results == i]
                if len(cluster_members) == 0:
                    family_names[i] = "0"
                else:
                    family_names[i] = get_family_name(cluster_members["Name"])
        return pd.DataFrame({
            "name": elements["Name"],
            "family1": [family_names[i] for i in results]
        })