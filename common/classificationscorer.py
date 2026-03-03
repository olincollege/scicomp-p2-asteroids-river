"""
Tools for scoring a given attempt at family classification.
"""

from dataclasses import dataclass
import pandas as pd
from sklearn import metrics
from typing import Tuple, Dict

@dataclass
class CarrieMeasureResult:
    """
    The result of computing the Carrie measure for a single predicted family and a true family.
    """
    corresponding_true_family: str # the true family that this result was computed relative to
    carrie_measure_pass: bool # whether this predicted family meets Carrie's criteria with respect to the corresponding true family
    true_positive_rate: float # the number of correctly identified family members divided by the total number of true family members
    false_positive_rate: float # the number of incorrectly identified family members divided by the total number of predicted family members
    

class ClassificationScorer:
    def __init__(self, predicted_labels: pd.DataFrame, true_labels: pd.DataFrame):
        """
        Initialize the scorer with the predicted labels.
        
        Parameters:
        predicted_labels (pd.DataFrame): A DataFrame containing the predicted family classification of the asteroids. 
            Columns: 'name', 'family1', both str. Family1 is "0" if the asteroid is not a member of any family,
            or the name of the family it is a member of.
        true_labels (pd.DataFrame): A DataFrame containing the true family classification of the asteroids.
            This is intended to be loaded directly from data/family_membership_*.csv.
            Columns: 'name', 'family1', both str. If an asteroid is not a member of any family, it is not included.
        """
        self.predicted_labels = predicted_labels
        self.true_labels = true_labels
        # insert missing rows into true_labels with family1=0 for asteroids that are not members of any family
        missing_asteroids = set(predicted_labels["name"]) - set(true_labels["name"])
        missing_rows = pd.DataFrame({"name": list(missing_asteroids), "family1": "0"})
        self.true_labels = pd.concat([true_labels, missing_rows], ignore_index=True)
    
    def rand_index(self) -> float:
        """
        Compute the chance-adjusted [Rand index](https://scikit-learn.org/stable/modules/clustering.html#rand-index) of the classification.
        
        Returns:
        float: The adjusted Rand index of the classification, between 0 and 1. Higher is better.
        """
        # merge the predicted and true labels on the name column
        merged = pd.merge(self.predicted_labels, self.true_labels, on="name", suffixes=("_pred", "_true"))
        # compute the adjusted Rand index using sklearn
        return metrics.adjusted_rand_score(merged["family1_true"], merged["family1_pred"])
    
    def v_measure(self) -> float:
        """
        Compute the [V-measure](https://scikit-learn.org/stable/modules/clustering.html#v-measure) of the classification.
        
        Returns:
        float: The V-measure of the classification, between 0 and 1. Higher is better.
        """
        # merge the predicted and true labels on the name column
        merged = pd.merge(self.predicted_labels, self.true_labels, on="name", suffixes=("_pred", "_true"))
        # compute the V-measure using sklearn
        return metrics.v_measure_score(merged["family1_true"], merged["family1_pred"])
    
    def _carrie_measure_single(self, predicted_family: str, true_family: str) -> CarrieMeasureResult:
        """
        Compute the Carrie measure for a single predicted family and a single true family.

        Parameters:
        predicted_family (str): The name of the predicted family to evaluate.
        true_family (str): The name of the true family to evaluate against.
        Returns:
        CarrieMeasureResult: The result of computing the Carrie measure for the given predicted and true family.
        """
        # get the set of asteroids in the predicted family
        predicted_members = set(self.predicted_labels[self.predicted_labels["family1"] == predicted_family]["name"])
        # get the set of asteroids in the true family
        true_members = set(self.true_labels[self.true_labels["family1"] == true_family]["name"])
        # compute the true positive rate and false positive rate
        true_positive_rate = len(predicted_members.intersection(true_members)) / len(true_members) if len(true_members) > 0 else 0
        false_positive_rate = len(predicted_members - true_members) / len(predicted_members) if len(predicted_members) > 0 else 0
        # compute whether this predicted family meets Carrie's criteria with respect to the true family
        carrie_measure_pass = true_positive_rate >= 0.95 and false_positive_rate <= 0.05
        return CarrieMeasureResult(
            corresponding_true_family=true_family,
            carrie_measure_pass=carrie_measure_pass,
            true_positive_rate=true_positive_rate,
            false_positive_rate=false_positive_rate
        )
    
    def carrie_measure(self) -> Tuple[int, Dict[str, CarrieMeasureResult]]:
        """
        Compute the Carrie measure, that is, Carrie's criteria for a successful project.
        More completely, the Carrie measure is the number of predicted families that
        contain at least 95% of the members of the corresponding true family, and also
        contain no more than 5% members that are *not* in the corresponding true family.
        The "corresponding true family" for a predicted family is the true family that
        scores the highest on the above criteria. The non-family asteroids (family1=0)
        are considered for the 5% non-member check, but not elsewhere.

        Returns:
        int: The Carrie measure of the classification, between 0 and the number of true families. Higher is better.
        dict: A dictionary mapping each predicted family to its CarrieMeasureResult
        """
        # get the set of true families
        true_families = set(self.true_labels["family1"])
        # compute the Carrie measure for each predicted family against each true family, and keep the best result for each predicted family
        carrie_results = {}
        for predicted_family in set(self.predicted_labels["family1"]):
            best_result = None
            for true_family in true_families:
                result = self._carrie_measure_single(predicted_family, true_family)
                if best_result is None or (result.true_positive_rate > best_result.true_positive_rate) or (result.true_positive_rate == best_result.true_positive_rate and result.false_positive_rate < best_result.false_positive_rate):
                    best_result = result
            carrie_results[predicted_family] = best_result
        # compute the Carrie measure as the number of predicted families that pass Carrie's criteria with respect to their corresponding true family
        carrie_measure_score = sum(1 for result in carrie_results.values() if result.carrie_measure_pass)
        return carrie_measure_score, carrie_results