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
from common.inputs import build_arg_parser, get_parameters
import numpy as np


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
                values = np.linspace(start, end, int((end - start) / step) + 1)
                swept_params.append((param_name, values))
            except ValueError:
                print(f"Invalid parameter sweep format for {param_name}: '{param_value}'. Expected format: start:end:step")
                return

    if not swept_params:
        print("No parameters to sweep. Please provide at least one parameter with a range (e.g. --param eps=0.01:0.6:0.01).")
        return
    
    print(f"Sweeping over parameters: {', '.join(f'{name}={values}' for name, values in swept_params)}")