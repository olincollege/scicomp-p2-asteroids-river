"""
Sweep over a range of parameter values for a chosen clustering algorithm.
Usage:
    python -m steps.20_parameter_sweep --dataset train --classifier dbscan --param eps=0.01:0.6:0.01 --param min_samples=5

    This will run DBSCAN with eps values from 0.01 to 0.6 in increments of 0.01, and min_samples fixed at 5.

    Input can also be given interactively.
"""

from common.classifiers import (
    get_classifier_by_name,
    choose_classifier_interactive,
)
from common.datasets import AVAILABLE_DATASETS, load_dataset, choose_dataset_interactive
from common.classificationscorer import ClassificationScorer
from common.familyclassifier import FamilyClassifier
from common.inputs import build_arg_parser, get_parameters
import numpy as np
import os

def get_result_string(parameters: dict, scorer: ClassificationScorer) -> str:
    """
    Get a string representation of the results that can be stored in a CSV file.
    Includes parameters for this run, number of predicted families, number of non-family asteroids, V-measure, and total Carrie measure.

    Parameters:
    parameters (dict): The dictionary of parameters used for this run.
    scorer (ClassificationScorer): The ClassificationScorer object containing the results to summarize.
    Returns:
    str: A string summarizing the results.
    """
    num_families = scorer.num_families()
    num_non_family = scorer.num_non_family_asteroids()
    v_measure = scorer.v_measure()
    total_carrie_measure, _ = scorer.carrie_measure()
    param_str = ", ".join(f"{value:.4f}" for value in parameters.values())
    return f"{param_str}, {num_families}, {num_non_family}, {v_measure:.4f}, {total_carrie_measure:.4f}"


def main():
    print("==== Parameter Sweep ====")

    parser = build_arg_parser("Sweep over a range of parameter values for a chosen clustering algorithm.")
    args, _ = parser.parse_known_args()

    # dataset
    dataset_name = args.dataset if args.dataset else choose_dataset_interactive()
    proper_elements, family_membership = load_dataset(dataset_name)
    print(f"Loaded dataset: {dataset_name}")

    # classifier
    classifier_name = args.classifier if args.classifier else choose_classifier_interactive()

    # parameters
    parameters = get_parameters(parser, args, classifier_name)
    swept_params = []
    for param_name, param_value in parameters.items():
        if param_value is not None and isinstance(param_value, str) and ":" in param_value:
            try:
                start, end, step = map(float, param_value.split(":"))
                values = np.linspace(start, end, int((end - start) / step), endpoint=False)
                swept_params.append((param_name, values))
            except ValueError:
                print(f"Invalid parameter sweep format for {param_name}: '{param_value}'. Expected format: start:end:step")
                return

    if not swept_params:
        print("No parameters to sweep. Please provide at least one parameter with a range (e.g. --param eps=0.01:0.6:0.01).")
        return
    
    print(f"Sweeping over parameters: {', '.join(f'{name}={values}' for name, values in swept_params)}")
    output_file = os.path.join("data", "sweeps", f"{classifier_name}_{dataset_name}_{'_'.join(f'{name}_{value.replace(":", "_")}' for name, value in parameters.items())}_results.csv")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, "w") as f:
        header = ", ".join(name for name, _ in swept_params) + ", num_families, num_non_family, v_measure, total_carrie_measure\n"
        f.write(header)
    # run the sweep over every combination of parameter values
    param_combinations = np.array(np.meshgrid(*[values for _, values in swept_params])).T.reshape(-1, len(swept_params))
    for combination in param_combinations:
        parameters = {name: value for (name, _), value in zip(swept_params, combination)}
        print(f"Running with parameters: {parameters}")
        classifier_cls = get_classifier_by_name(classifier_name)
        classifier = classifier_cls(**parameters)
        predicted_labels = classifier.classify(proper_elements)
        scorer = ClassificationScorer(predicted_labels, family_membership)
        result_string = get_result_string(parameters, scorer)
        with open(output_file, "a") as f:
            f.write(result_string + "\n")
        print(f"Results: {result_string}")

if __name__ == "__main__":
    main()