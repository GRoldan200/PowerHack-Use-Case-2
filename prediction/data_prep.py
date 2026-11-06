"""Loads and cleans the training and testing data shared by the prediction scripts."""
import pandas as pd
import numpy as np

# Separate files with no buildings in common, so the test scores are honest
training_df = pd.read_csv("../multifamily_building_energy_consumption_training_dataset.csv")
test_df = pd.read_csv("../multifamily_building_energy_consumption_testing_dataset.csv")

energy_col = "Weather Normalized Site Energy Use (kBtu)"
area_col = "Largest Property Use Type - Gross Floor Area (ft^2)"

# The testing CSV's header was saved as "(ft?)" instead of "(ft^2)"
test_df = test_df.rename(columns={
    "Largest Property Use Type - Gross Floor Area (ft?)": area_col
})

# Calculate EUI
for df in [training_df, test_df]:
    df["EUI"] = df[energy_col] / df[area_col].replace(0, np.nan)

# Remove invalid target values
training_df = training_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

training_df = training_df.dropna(subset=["EUI"])
test_df = test_df.dropna(subset=["EUI"])

# Missing population type is "Not Available" in training and "0" in testing,
# so group both (and blanks) into one "Unknown" category
population_col = "Multifamily Housing - Resident Population Type"
for df in [training_df, test_df]:
    df[population_col] = (
        df[population_col]
        .replace({"0": np.nan, "Not Available": np.nan})
        .fillna("Unknown")
    )

# The testing CSV stores "Not Available" laundry hookups as 0, mixed in with real zeros,
# so code the training CSV the same way to keep both files consistent
laundry_cols = [
    "Multifamily Housing - Number of Laundry Hookups in All Units",
    "Multifamily Housing - Number of Laundry Hookups in Common Area(s)"
]
for df in [training_df, test_df]:
    for col in laundry_cols:
        df[col] = pd.to_numeric(df[col].replace("Not Available", "0"), errors="coerce")

# Drop implausible EUI values (likely data errors) from training only;
# the test set stays as-is so its scores reflect real-world data
training_df = training_df[training_df["EUI"].between(5, 400)]

target = "EUI"

features = [
    "Year Built",
    "Occupancy",
    "Multifamily Housing - Total Number of Residential Living Units",
    "Multifamily Housing - Number of Bedrooms",
    "Multifamily Housing - Percent That Can Be Heated",
    "Multifamily Housing - Percent That Can Be Cooled",
    "Multifamily Housing - Number of Laundry Hookups in All Units",
    "Multifamily Housing - Number of Laundry Hookups in Common Area(s)",
    "Multifamily Housing - Number of Residential Living Units in a High-Rise Building (10 or more Stories)",
    "Multifamily Housing - Number of Residential Living Units in a Mid-Rise Building (5-9 Stories)",
    # Densities divide counts by floor area; floor area is known before energy use,
    # so these aren't leakage and they are the strongest predictors available
    "Multifamily Housing - Number of Bedrooms Density (Number per 1,000 sq ft)",
    "Multifamily Housing - Total Number of Residential Living Units Density (Number per 1,000 sq ft)",
    "Multifamily Housing - Number of Residential Living Units in a High-Rise Building Density (Number per 1,000 sq ft)",
    "Multifamily Housing - Number of Residential Living Units in a Mid-Rise Building Density (Number per 1,000 sq ft)",
    "Multifamily Housing - Resident Population Type"
]

X_train = training_df[features]
y_train = training_df[target]

X_test = test_df[features]
y_test = test_df[target]

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns

# Keep test values inside the training range, e.g. one test building
# lists 100,000 laundry hookups, which sends linear models to -2,500 EUI
X_test = X_test.copy()
X_test[numeric_features] = X_test[numeric_features].clip(
    lower=X_train[numeric_features].min(),
    upper=X_train[numeric_features].max(),
    axis=1
)

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns
