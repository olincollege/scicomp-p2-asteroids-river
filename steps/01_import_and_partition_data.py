# Asteroid classification
# 
# The data is from [AstDys](https://newton.spacedys.com/astdys2/index.php?pc=5),
# datasets "Numbered and multiopposition asteroids" and "Individual asteroid family membership."

# Load the data:
import pandas as pd
import numpy as np
family_membership_all = pd.read_csv("data/all_tro.members.txt", sep=r"\s+", comment="%", low_memory=False)
proper_elements = pd.read_csv("data/proper_elements.txt", sep=r"\s+", comment="%", low_memory=False)

# Check that the data got imported correctly
print("Proper elements:")
print(proper_elements.head())

print("Family membership:")
print(family_membership_all.head())


# Split the family membership data into training, test, and validation sets. 
# To do this, we'll get all the families that exist, then randomly split them into 3 groups, 
# then cross-reference asteroid elements into training/test/validation sets to match.

families = pd.unique(family_membership_all["family1"])
rng = np.random.default_rng(seed=42)
rng.shuffle(families) #type: ignore # the type checker is scared of families being an ExtensionArray instead of an np_1darray but that never happens in practice
[families_train_raw, families_test_raw, families_validate_raw] = np.array_split(families, 3) #type: ignore # see above
# convert from numpy to dataframe
families_train = pd.DataFrame(families_train_raw, columns=["family1"])
families_test = pd.DataFrame(families_test_raw, columns=["family1"])
families_validate = pd.DataFrame(families_validate_raw, columns=["family1"])
# reconstruct family_membership_[train|test|validate]
family_membership_train = family_membership_all[family_membership_all["family1"].isin(families_train["family1"])]
family_membership_test = family_membership_all[family_membership_all["family1"].isin(families_test["family1"])]
family_membership_validate = family_membership_all[family_membership_all["family1"].isin(families_validate["family1"])]

# split the proper elements into training, test, and validation sets based on the family membership sets.
proper_elements_train = proper_elements[proper_elements["Name"].isin(family_membership_train["name"])]
proper_elements_test = proper_elements[proper_elements["Name"].isin(family_membership_test["name"])]
proper_elements_validate = proper_elements[proper_elements["Name"].isin(family_membership_validate["name"])]
proper_elements_no_family = proper_elements[~proper_elements["Name"].isin(family_membership_all["name"])].sample(frac=1, random_state=rng)
# split up the no-family asteroids into train, test, and validate as well so there's some representation of them in each set
[proper_elements_no_family_train_raw, proper_elements_no_family_test_raw, proper_elements_no_family_validate_raw] = np.array_split(proper_elements_no_family, 3)
proper_elements_no_family_train = pd.DataFrame(proper_elements_no_family_train_raw, columns=proper_elements.columns)
proper_elements_no_family_test = pd.DataFrame(proper_elements_no_family_test_raw, columns=proper_elements.columns)
proper_elements_no_family_validate = pd.DataFrame(proper_elements_no_family_validate_raw, columns=proper_elements.columns)
proper_elements_train = pd.concat([proper_elements_train, proper_elements_no_family_train])
proper_elements_test = pd.concat([proper_elements_test, proper_elements_no_family_test])
proper_elements_validate = pd.concat([proper_elements_validate, proper_elements_no_family_validate])

# Validation

# check that the proper element sets are disjoint
assert set(proper_elements_train["Name"]).isdisjoint(set(proper_elements_test["Name"]))
assert set(proper_elements_train["Name"]).isdisjoint(set(proper_elements_validate["Name"]))
assert set(proper_elements_test["Name"]).isdisjoint(set(proper_elements_validate["Name"]))
# check that the proper element sets sum to the original set
assert len(proper_elements_train) + len(proper_elements_test) + len(proper_elements_validate) == len(proper_elements)
# check that the proper elements include both family and no-family asteroids
assert len(set(proper_elements_train["Name"]).intersection(set(family_membership_train["name"]))) > 0, "proper_elements_train should include some family asteroids"
assert len(set(proper_elements_train["Name"]).intersection(set(family_membership_train["name"]))) < len(proper_elements_train), "proper_elements_train should include some no-family asteroids"
assert len(set(proper_elements_test["Name"]).intersection(set(family_membership_test["name"]))) > 0, "proper_elements_test should include some family asteroids"
assert len(set(proper_elements_test["Name"]).intersection(set(family_membership_test["name"]))) < len(proper_elements_test), "proper_elements_test should include some no-family asteroids"
assert len(set(proper_elements_validate["Name"]).intersection(set(family_membership_validate["name"]))) > 0, "proper_elements_validate should include some family asteroids"
assert len(set(proper_elements_validate["Name"]).intersection(set(family_membership_validate["name"]))) < len(proper_elements_validate), "proper_elements_validate should include some no-family asteroids"
# check that the family membership sets are disjoint
assert set(family_membership_train["name"]).isdisjoint(set(family_membership_test["name"]))
assert set(family_membership_train["name"]).isdisjoint(set(family_membership_validate["name"]))
assert set(family_membership_test["name"]).isdisjoint(set(family_membership_validate["name"]))
# check that the family membership sets sum to the original set
assert len(family_membership_train) + len(family_membership_test) + len(family_membership_validate) == len(family_membership_all)
# check that no families are split across sets
assert set(family_membership_train["family1"]).isdisjoint(set(family_membership_test["family1"])), "a family is split across train and test"
assert set(family_membership_train["family1"]).isdisjoint(set(family_membership_validate["family1"])), "a family is split across train and validate"
assert set(family_membership_test["family1"]).isdisjoint(set(family_membership_validate["family1"])), "a family is split across test and validate"
print("All checks passed!")

# print some summary statistics and samples about the resulting sets 
print(f"Number of families in each set: train={len(families_train)}, test={len(families_test)}, validate={len(families_validate)}")
print("Family membership train set size:", len(family_membership_train))
print("Family membership test set size:", len(family_membership_test))
print("Family membership validate set size:", len(family_membership_validate))
print("Proper elements train set size:", len(proper_elements_train))
print("Proper elements test set size:", len(proper_elements_test))
print("Proper elements validate set size:", len(proper_elements_validate))
print("Family membership train set sample:")
print(family_membership_train.sample(5, random_state=rng))
print("Proper elements train set sample:")
print(proper_elements_train.sample(5, random_state=rng))
print("Family membership test set sample:")
print(family_membership_test.sample(5, random_state=rng))
print("Family membership validate set sample:")
print(family_membership_validate.sample(5, random_state=rng))

# save the data
family_membership_train.to_csv("data/family_membership_train.csv", index=False)
family_membership_test.to_csv("data/family_membership_test.csv", index=False)
family_membership_validate.to_csv("data/family_membership_validate.csv", index=False)
proper_elements_train.to_csv("data/proper_elements_train.csv", index=False)
proper_elements_test.to_csv("data/proper_elements_test.csv", index=False)
proper_elements_validate.to_csv("data/proper_elements_validate.csv", index=False)