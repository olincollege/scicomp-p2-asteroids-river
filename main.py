import pandas as pd

def import_data():
    family_membership = pd.read_csv("all_tro.members.txt")
    proper_elements = pd.read_csv("proper_elements.csv")

def main():
    import_data()


if __name__ == "__main__":
    main()
