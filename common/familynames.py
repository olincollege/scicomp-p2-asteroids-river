"""
Helper for sorting names of asteroids.
"""
from numpy.typing import ArrayLike

def get_family_name(member_names: set | ArrayLike) -> str:
    """
    Get the name of a family given the set of its member names. The name of a family is the name of its earliest-sorted member.
    Members are sorted in the following order:
    First, all numeric names, ascending.
    Then, all non-numeric names, sorted alphabetically.
    This puts 108 before 322542 before 2008RW68 before 2015HF252.

    Parameters:
    member_names (set or list of str): The set of names of the members of the family.
    Returns:
    str: The name of the family.
    """
    numeric_names = sorted([name for name in member_names if name.isdigit()], key=int)
    non_numeric_names = sorted([name for name in member_names if not name.isdigit()])
    sorted_names = numeric_names + non_numeric_names
    return sorted_names[0] if len(sorted_names) > 0 else "0"