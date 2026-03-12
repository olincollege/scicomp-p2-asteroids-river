"""
Helpers for loading and selecting datasets.
"""

import pandas as pd
from typing import List, Tuple

AVAILABLE_DATASETS = ["train", "test", "validate", "all"]

def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the proper elements and family membership data for the given dataset.

    Parameters:
    name (str): One of "train", "test", "validate", or "all".

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: (proper_elements, family_membership)
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(AVAILABLE_DATASETS)}")
    if name == "all":
        family_membership = pd.read_csv("data/all_tro.members.txt", sep=r"\s+", comment="%", dtype={"name": str, "family1": str, "near1": str})
        proper_elements = pd.read_csv("data/proper_elements.txt", sep=r"\s+", comment="%", dtype={"Name": str})
    else:
        proper_elements = pd.read_csv(f"data/proper_elements_{name}.csv", dtype={"Name": str})
        family_membership = pd.read_csv(f"data/family_membership_{name}.csv", dtype={"name": str, "family1": str, "near1": str})
    return proper_elements, family_membership


def choose_dataset_interactive() -> str:
    """
    Prompt the user to pick a dataset interactively.

    Returns:
    str: The chosen dataset name (e.g. "train").
    """
    print("\nAvailable datasets:")
    for i, name in enumerate(AVAILABLE_DATASETS, 1):
        print(f"  {i}. {name}")
    raw = input("Enter the number of the dataset to use: ")
    try:
        idx = int(raw)
        if idx < 1 or idx > len(AVAILABLE_DATASETS):
            raise ValueError()
    except ValueError:
        raise SystemExit("Invalid dataset choice, exiting.")
    return AVAILABLE_DATASETS[idx - 1]

def choose_datasets_interactive() -> List[str]:
    """
    Prompt the user to pick one or more datasets interactively.

    Returns:
    List[str]: The chosen dataset names (e.g. ["train", "validate"]).
    """
    print("\nAvailable datasets:")
    for i, name in enumerate(AVAILABLE_DATASETS, 1):
        print(f"  {i}. {name}")
    raw = input("Enter the numbers of the datasets to use, separated by commas (e.g. '1,3' for train and validate): ")
    try:
        idxs = [int(x.strip()) for x in raw.split(",")]
        if any(idx < 1 or idx > len(AVAILABLE_DATASETS) for idx in idxs):
            raise ValueError()
    except ValueError:
        raise SystemExit("Invalid dataset choice, exiting.")
    return [AVAILABLE_DATASETS[idx - 1] for idx in idxs]