"""
Helpers for loading and selecting datasets.
"""

import pandas as pd
from typing import Tuple

AVAILABLE_DATASETS = ["train", "test", "validate"]

def load_dataset(name: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the proper elements and family membership data for the given dataset.

    Parameters:
    name (str): One of "train", "test", or "validate".

    Returns:
    Tuple[pd.DataFrame, pd.DataFrame]: (proper_elements, family_membership)
    """
    if name not in AVAILABLE_DATASETS:
        raise ValueError(f"Unknown dataset '{name}'. Choose from: {', '.join(AVAILABLE_DATASETS)}")
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
