from common.familyclassifier import FamilyClassifier

from . import kmeans, dbscan, hdbscan, dbscan_3param, dbscan_3param_norm


all_classifiers: dict[str, type[FamilyClassifier]] = {
    "kmeans": kmeans.KMeansFamilyClassifier,
    "dbscan": dbscan.DBSCANFamilyClassifier,
    "dbscan_3param": dbscan_3param.DBSCAN3FamilyClassifier,
    "dbscan_3param_norm": dbscan_3param_norm.DBSCAN3NormFamilyClassifier,
    "hdbscan": hdbscan.HDBSCANFamilyClassifier,
}


def get_classifier_by_name(name: str) -> type[FamilyClassifier]:
    """
    Look up a classifier class by name (case-insensitive).

    Parameters:
    name (str): The name of the classifier (e.g. "dbscan").

    Returns:
    The classifier class.

    Raises:
    SystemExit: If the name is not recognised.
    """
    key = name.lower()
    if key not in all_classifiers:
        raise SystemExit(
            f"Unknown classifier '{name}'. Available: {', '.join(all_classifiers.keys())}"
        )
    return all_classifiers[key]


def choose_classifier_interactive() -> str:
    """
    Prompt the user to pick a classifier interactively.

    Returns:
    str: The chosen classifier name (a key in ``all_classifiers``).
    """
    print("\nAvailable classifiers:")
    names = list(all_classifiers.keys())
    for i, name in enumerate(names, 1):
        print(f"  {i}. {name}")
    raw = input("Enter the number of the classifier to use: ")
    try:
        idx = int(raw)
        if idx < 1 or idx > len(names):
            raise ValueError()
    except ValueError:
        raise SystemExit("Invalid classifier choice, exiting.")
    return names[idx - 1]


def collect_params_interactive(classifier_name: str) -> dict:
    """
    Interactively collect parameter values for the given classifier.

    Parameters:
    classifier_name (str): A key in ``all_classifiers``.

    Returns:
    dict: The parameter values the user entered (defaults omitted).
    """
    cls = get_classifier_by_name(classifier_name)
    parameters = cls.get_params()
    print(f"\nSelected classifier: {classifier_name}")
    for param_name, param_desc in parameters.items():
        print(f"  Parameter: {param_name}")
        print(f"  Description: {param_desc}")
        value = input(f"  Enter a value for {param_name} (or press enter for default): ")
        if value == "":
            parameters[param_name] = None
        else:
            try:
                value = int(value)
            except ValueError:
                try:
                    value = float(value)
                except ValueError:
                    pass
            parameters[param_name] = value
    return {k: v for k, v in parameters.items() if v is not None}


def parse_param_value(value_str: str):
    """
    Try to convert a string parameter value to int, then float, falling back to str.
    """
    try:
        return int(value_str)
    except ValueError:
        try:
            return float(value_str)
        except ValueError:
            return value_str