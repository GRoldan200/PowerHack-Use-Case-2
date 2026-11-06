import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression

training_df = pd.read_csv("../multi_family_eui.csv")
test_df = pd.read_csv("../multifamily_building_energy_consumption_testing_dataset.csv")

energy_col = "Weather Normalized Site Energy Use (kBtu)"
area_col = "Largest Property Use Type - Gross Floor Area (ft²)"

# Calculate EUI
for df in [training_df, test_df]:
    df["EUI"] = df[energy_col] / df[area_col].replace(0, np.nan)

# Remove invalid target values
training_df = training_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

training_df = training_df.dropna(subset=["EUI"])
test_df = test_df.dropna(subset=["EUI"])


# Inspect dataset
# print(training_df.head())
# print(training_df.info())
# print(training_df.describe())

# print(training_df.columns.tolist())
# print(test_df.columns.tolist())

# exit(0)

target = "EUI"

features = [
    "Year Built",
    "Occupancy",
    "Largest Property Use Type - Gross Floor Area (ft²)",
    "Multifamily Housing - Total Number of Residential Living Units",
    "Multifamily Housing - Number of Bedrooms",
    "Multifamily Housing - Percent That Can Be Heated",
    "Multifamily Housing - Percent That Can Be Cooled",
    "Multifamily Housing - Resident Population Type"
]

X_train = training_df[features]
y_train = training_df[target]

X_test = test_df[features]
y_test = test_df[target]

numeric_features = X_train.select_dtypes(
    include=["number"]
).columns

categorical_features = X_train.select_dtypes(
    include=["object", "category", "bool"]
).columns

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", categorical_pipeline, categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", LinearRegression())
])

model.fit(X_train, y_train)

# Predictions
predictions = model.predict(X_test)

# Evaluate the model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

mse = mean_squared_error(y_test, predictions)
r2 = r2_score(y_test, predictions)
mae = mean_absolute_error(y_test, predictions)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")
print(f"Mean Absolute Error: {mae}")


import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

plt.scatter(y_test, predictions, alpha=0.5)

# Perfect prediction reference line
min_val = min(y_test.min(), predictions.min())
max_val = max(y_test.max(), predictions.max())

plt.plot(
    [min_val, max_val],
    [min_val, max_val],
    linestyle="--"
)

plt.xlabel("Actual EUI (kBtu/ft²)")
plt.ylabel("Predicted EUI (kBtu/ft²)")
plt.title("Actual vs Predicted Building EUI")

plt.show()