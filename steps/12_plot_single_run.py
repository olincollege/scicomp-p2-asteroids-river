"""
Do a single classification run, then plot the results.
The final plot is two subplots, both plotting asteroids
in the data set, a against sinI. Both plots show the families
that passed the Carrie measure in color - one shows the predicted families,
and the other shows the corresponding true families.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.markers import MarkerStyle
from matplotlib.patches import Rectangle
import numpy as np

from common.classificationscorer import ClassificationScorer
from common.classifiers import choose_classifier_interactive, get_classifier_by_name, collect_params_interactive
from common.datasets import choose_datasets_interactive, load_dataset

def main():
    # dataset
    dataset_names = choose_datasets_interactive()
    # classifier
    classifier_name = choose_classifier_interactive()
    parameters = collect_params_interactive(classifier_name)
    classifier_cls = get_classifier_by_name(classifier_name)
    classifier = classifier_cls(**parameters)
    families_passing_carrie = {}
    all_proper_elements = None
    all_family_membership = None
    scorers = []
    for dataset_name in dataset_names:
        proper_elements, family_membership = load_dataset(dataset_name)
        all_proper_elements = proper_elements if all_proper_elements is None else pd.concat([all_proper_elements, proper_elements], ignore_index=True)
        all_family_membership = family_membership if all_family_membership is None else pd.concat([all_family_membership, family_membership], ignore_index=True)

        print(f"Running {classifier_name} on {dataset_name}...")
        results = classifier.classify(proper_elements)
        print("Classification results:")
        print(results.head())
        # score
        scorer = ClassificationScorer(results, family_membership)
        scorers.append(scorer)
        print(f"V-measure: {scorer.v_measure():.4f}")
        carrie_index_total, carrie_index_indiv = scorer.carrie_measure()
        print(f"Carrie index for this dataset(total): {carrie_index_total:.4f}")
        families_passing_carrie.update({fam: res for fam, res in carrie_index_indiv.items() if res.carrie_measure_pass})
        print(f"Families now passing Carrie measure: {len(families_passing_carrie)}")
    assert all_proper_elements is not None and all_family_membership is not None, "No datasets loaded, cannot plot results."
    # plot the results
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle(f"Classification results for {classifier_name} on {dataset_names}\n{parameters}")
    style = MarkerStyle("+").scaled(0.3)
    # plot all asteroids in dataset in light/translucent gray
    for ax in axs:
        ax.scatter(all_proper_elements["a"], all_proper_elements["sinI"], color="lightgray", alpha=0.2, label="Families", marker=style)
        ax.set_xlabel("a (semimajor axis)")
        ax.set_ylabel("sinI (inclination)")
        ax.set_xlim(2, 3.5)
        ax.set_ylim(0, 0.34) # 0 to sind(20)
    # color the ones in families that passed the Carrie measure by their family
    for fam, res in families_passing_carrie.items():
        pred_fam_members = all_family_membership[all_family_membership["family1"] == fam]["name"]
        pred_fam_proper_elements = all_proper_elements[all_proper_elements["Name"].isin(pred_fam_members)]
        axs[0].scatter(pred_fam_proper_elements["a"], pred_fam_proper_elements["sinI"], label=fam, marker=style)
        true_fam_members = all_family_membership[all_family_membership["family1"] == res.corresponding_true_family]["name"]
        true_fam_proper_elements = all_proper_elements[all_proper_elements["Name"].isin(true_fam_members)]
        axs[1].scatter(true_fam_proper_elements["a"], true_fam_proper_elements["sinI"], label=res.corresponding_true_family, marker=style)
    axs[0].set_title("Predicted families passing Carrie measure")
    axs[0].legend(loc="upper left", markerscale=3.5)
    axs[1].set_xlabel("a (semimajor axis)")
    axs[1].set_ylabel("sinI (inclination)")
    axs[1].set_title("Corresponding true families")
    axs[1].legend(loc="upper left", markerscale=3.5)
    plt.tight_layout()
    plt.show()

    # plot the carrie measure based on adjusting the false positive and true positive thresholds
    true_thresholds = np.linspace(0.85, 1.0, 15)
    false_thresholds = np.linspace(0.0, 0.15, 15)
    carrie_results_grid = np.zeros((len(true_thresholds), len(false_thresholds)))
    for i, true_threshold in enumerate(true_thresholds):
        for j, false_threshold in enumerate(false_thresholds):
            carrie_index_totals = 0
            for scorer in scorers:
                carrie_index_total, carrie_index_indiv = scorer.reinterpret_carrie_thresholds(true_pass_threshold=true_threshold, false_pass_threshold=false_threshold)
                carrie_index_totals += carrie_index_total
            carrie_results_grid[i, j] = carrie_index_totals
    plt.figure(figsize=(8, 6))
    plt.imshow(carrie_results_grid, extent=(false_thresholds[0], false_thresholds[-1], true_thresholds[0], true_thresholds[-1]), aspect='auto', origin='lower', cmap='inferno')
    plt.colorbar(label="Carrie index")
    # put a box around the default thresholds of 0.95 true positive and 0.05 false positive
    patch = Rectangle((0.05, 0.95), 0.01, 0.01, fill=False, edgecolor='yellow', linewidth=2)
    plt.gca().add_patch(patch)
    plt.xlabel("False positive rate threshold")
    plt.ylabel("True positive rate threshold")
    plt.title("Carrie index for different true/false positive thresholds")
    plt.show()

if __name__ == "__main__":
    main() 
