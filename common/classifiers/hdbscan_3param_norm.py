from sklearn import cluster
from common.familyclassifier import FamilyClassifier
import pandas as pd
from common.familynames import get_family_name

class HDBSCAN3NormFamilyClassifier(FamilyClassifier):
    """
    HDBSCAN based asteroid family classifier. Only uses the proper elements (a, e, sinI) for clustering, and normalizes the data before clustering.
    """
    def __init__(self, min_cluster_size: int = 5, min_samples: int = None, cluster_selection_epsilon: float = 0.0, n_jobs: int = -1):
        """
        Initialize the HDBSCAN3NormFamilyClassifier with the given parameters.
        
        Parameters:
        min_cluster_size (int): The minimum size of clusters. Default is 5, which is a common choice for HDBSCAN.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point. Default is None, which means it will be set to the same value as min_cluster_size.
        cluster_selection_epsilon (float): The distance threshold for cluster selection. Default is 0.0, which means no distance threshold will be applied.
        n_jobs (int): The number of parallel jobs to run for HDBSCAN. Default is -1, which means using all available processors.
        """
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples) if min_samples is not None else self.min_cluster_size
        self.cluster_selection_epsilon = float(cluster_selection_epsilon)
        self.n_jobs = int(n_jobs)
        self.hdbscan = cluster.HDBSCAN(min_cluster_size=self.min_cluster_size, min_samples=self.min_samples, cluster_selection_epsilon=self.cluster_selection_epsilon, n_jobs=self.n_jobs, copy=False)

    @staticmethod
    def get_params() -> dict:
        """
        Get the parameters of the classifier as a dictionary. The keys should be the parameter names, and the values 
        should be a description of the parameter.
        
        Returns:
        dict: A dictionary containing the parameters of the classifier.
        """
        return {
            "min_cluster_size": "The minimum size of clusters. Default is 5, which is a common choice for HDBSCAN.",
            "min_samples": "The number of samples in a neighborhood for a point to be considered as a core point. Default is None, which means it will be set to the same value as min_cluster_size.",
            "cluster_selection_epsilon": "The distance threshold for cluster selection. Default is 0.0, which means no distance threshold will be applied.",
            "n_jobs": "The number of parallel jobs to run for HDBSCAN. Default is -1, which means using all available processors."
        }
    
    def classify(self, elements: pd.DataFrame) -> pd.DataFrame:
        """
        Classify the given asteroids and their proper elements into families using HDBSCAN clustering.
        See FamilyClassifier.classify for detailed description of the parameters and return value.
        
        Parameters:
        elements (pd.DataFrame): A DataFrame containing the proper elements of asteroids. 
        
        Returns:
        pd.DataFrame: A DataFrame containing the family classification of the asteroids. 
            
        """
        print("Normalizing data...")
        data_normalized = (elements[["a", "e", "sinI"]] - elements[["a", "e", "sinI"]].mean()) / elements[["a", "e", "sinI"]].std()
        print("Running HDBSCAN...")
        results = self.hdbscan.fit_predict(data_normalized)
        print(f"HDBSCAN found {len(set(results)) - (1 if -1 in results else 0)} clusters and {list(results).count(-1)} outliers.")
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