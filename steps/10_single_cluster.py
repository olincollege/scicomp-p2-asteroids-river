"""
Classify the data with a single clusterer.

Can be run interactively (no arguments) or fully from the command line:

    python -m steps.10_single_cluster --dataset train --classifier dbscan --param eps=0.05 --param min_samples=10
"""

from common.classifiers import (
    get_classifier_by_name,
    choose_classifier_interactive,
)
from common.datasets import AVAILABLE_DATASETS, load_dataset, choose_dataset_interactive
from common.classificationscorer import ClassificationScorer
from common.inputs import build_arg_parser, get_parameters


def main():
    print("==== Single Clusterer run ====")

    parser = build_arg_parser("Run a single clustering classifier on an asteroid dataset.")
    args, _ = parser.parse_known_args()

    # dataset
    dataset_name = args.dataset if args.dataset else choose_dataset_interactive()
    proper_elements, family_membership = load_dataset(dataset_name)
    print(f"Loaded dataset: {dataset_name}")

    # classifier
    classifier_name = args.classifier if args.classifier else choose_classifier_interactive()

    # parameters
    parameters = get_parameters(parser, args, classifier_name)

    # run
    classifier_cls = get_classifier_by_name(classifier_name)
    classifier = classifier_cls(**parameters)
    results = classifier.classify(proper_elements)

    print("Classification results:")
    print(results.head())

    # score
    scorer = ClassificationScorer(results, family_membership)
    print(f"V-measure: {scorer.v_measure():.4f}")

    carrie_index_total, carrie_index_indiv = scorer.carrie_measure()
    print(f"Carrie index (total): {carrie_index_total:.4f}")

    best_carrie_index_indiv = scorer.best_carrie_measure_individual(carrie_index_indiv)

    if best_carrie_index_indiv:
        fam, res = best_carrie_index_indiv
        print(
            f"Best Carrie index (individual) {fam}: "
            f"corresponding true family={res.corresponding_true_family}, "
            f"pass={res.carrie_measure_pass}, "
            f"true positive rate={res.true_positive_rate:.4f}, "
            f"false positive rate={res.false_positive_rate:.4f}"
        )


if __name__ == "__main__":
    main()
