"""
Download the required datasets for the project, and mutate them slightly to allow easier importing.
"""

import requests

def main():
    print("=== Downloading datasets from AstDys... ===")
    family_membership_url = "https://newton.spacedys.com/~astdys2/propsynth/all_tro.members"
    proper_elements_url = "https://newton.spacedys.com/~astdys2/propsynth/all.syn"
    family_membership_headers = "name       Hmag   status family1      dv_fam1 near1   family2     dv_fam2 near2    rescod"
    proper_elements_headers = "Name       mag     a            e       sinI             n        g                s  LCEx1E6  My"

    print("Downloading family membership data...")
    family_membership_response = requests.get(family_membership_url)
    print(f"Downloaded {len(family_membership_response.content) / (1024 * 1024):.2f} MB of family membership data, saving to data/all_tro.members.txt...")
    with open("data/all_tro.members.txt", "w") as f:
        f.write(family_membership_headers + "\n" + family_membership_response.text)
    
    print("Downloading proper elements data...")
    proper_elements_response = requests.get(proper_elements_url)
    print(f"Downloaded {len(proper_elements_response.content) / (1024 * 1024):.2f} MB of proper elements data, saving to data/proper_elements.txt...")
    with open("data/proper_elements.txt", "w") as f:
        f.write(proper_elements_headers + "\n" + proper_elements_response.text)
    print("All datasets downloaded.")

if __name__ == "__main__":
    main()