
import argparse

from common import classifiers
from common.datasets import AVAILABLE_DATASETS

def build_arg_parser(desc: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=desc
    )
    parser.add_argument(
        "--dataset",
        choices=AVAILABLE_DATASETS,
        default=None,
        help="Dataset to use (train, test, validate). Prompts interactively if omitted.",
    )
    parser.add_argument(
        "--classifier",
        choices=list(classifiers.all_classifiers.keys()),
        default=None,
        help="Classifier to use. Prompts interactively if omitted.",
    )
    parser.add_argument(
        "--param",
        action="append",
        metavar="KEY=VALUE",
        default=[],
        help="Classifier parameter as key=value. Can be repeated. "
             "Prompts interactively if no --param flags are given and --classifier is also omitted.",
    )
    return parser

def get_parameters(parser: argparse.ArgumentParser, args: argparse.Namespace, classifier_name: str) -> dict:
    if args.param:
        # Parse key=value pairs from the command line 
        parameters = {}
        for item in args.param:
            if "=" not in item:
                parser.error(f"--param values must be in KEY=VALUE format, got '{item}'")
            key, value = item.split("=", 1)
            parameters[key] = classifiers.parse_param_value(value)
    elif args.classifier:
        # Classifier given on CLI but no params → use defaults
        parameters = {}
    else:
        # Fully interactive
        parameters = classifiers.collect_params_interactive(classifier_name)
    return parameters