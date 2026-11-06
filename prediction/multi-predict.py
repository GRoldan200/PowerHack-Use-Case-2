import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import (
    RandomForestRegressor,
    GradientBoostingRegressor
)
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

training_df = pd.read_csv("../multi_family_eui.csv")
test_df = pd.read_csv("../multifamily_building_energy_consumption_testing_dataset.csv")

energy_col = "Weather Normalized Site Energy Use (kBtu)"
area_col = "Largest Property Use Type - Gross Floor Area (ft^2)"

# Calculate EUI
for df in [training_df, test_df]:
    df["EUI"] = df[energy_col] / df[area_col].replace(0, np.nan)

# Remove invalid target values
training_df = training_df.replace([np.inf, -np.inf], np.nan)
test_df = test_df.replace([np.inf, -np.inf], np.nan)

training_df = training_df.dropna(subset=["EUI"])
test_df = test_df.dropna(subset=["EUI"])

target = "EUI"

features = [
    "Year Built",
    "Occupancy",
    "Largest Property Use Type - Gross Floor Area (ft^2)",
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

models = {
    # Baseline: predict the median training EUI for every building
    # (same as floor area × median EUI, once divided by floor area)
    "Baseline (median EUI)": DummyRegressor(strategy="median"),

    "Linear Regression": LinearRegression(),

    "Ridge Regression": Ridge(alpha=1.0),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    ),

    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    )
}

results = []
all_predictions = {}
fitted_pipelines = {}

for name, regressor in models.items():

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor)
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)
    all_predictions[name] = predictions
    fitted_pipelines[name] = pipeline

    results.append({
        "Model": name,
        "MAE": mean_absolute_error(y_test, predictions),
        "RMSE": mean_squared_error(
            y_test, predictions
        ) ** 0.5,
        "R2": r2_score(y_test, predictions)
    })

results_df = pd.DataFrame(results)

print(results_df.sort_values("RMSE"))


# P-values for the linear regression coefficients (OLS t-tests)
# Drop the first one-hot category so the dummies aren't collinear with the intercept
stats_preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(
            drop="first",
            handle_unknown="ignore",
            sparse_output=False
        ))
    ]), categorical_features)
])

X_design = stats_preprocessor.fit_transform(X_train)
X_design = np.column_stack([np.ones(len(X_design)), X_design])
feature_names = ["Intercept"] + list(stats_preprocessor.get_feature_names_out())

y = y_train.to_numpy()
coefficients, _, _, _ = np.linalg.lstsq(X_design, y, rcond=None)

residuals = y - X_design @ coefficients
n, k = X_design.shape
degrees_of_freedom = n - k
residual_variance = residuals @ residuals / degrees_of_freedom

covariance = residual_variance * np.linalg.pinv(X_design.T @ X_design)
standard_errors = np.sqrt(np.diag(covariance))
t_values = coefficients / standard_errors
p_values = 2 * stats.t.sf(np.abs(t_values), degrees_of_freedom)

p_values_df = pd.DataFrame({
    "Feature": feature_names,
    "Coefficient": coefficients,
    "Std Error": standard_errors,
    "t": t_values,
    "p-value": p_values
})

print()
print("Linear regression p-values (training data, scaled features):")
print(p_values_df.sort_values("p-value").to_string(index=False))


# Charts
SERIES_COLOR = "#2a78d6"
MUTED_COLOR = "#b8b7b0"
TEXT_COLOR = "#0b0b0b"
SECONDARY_TEXT = "#52514e"
GRID_COLOR = "#e4e3de"

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb",
    "axes.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID_COLOR,
    "axes.labelcolor": SECONDARY_TEXT,
    "axes.titlecolor": TEXT_COLOR,
    "xtick.color": SECONDARY_TEXT,
    "ytick.color": SECONDARY_TEXT,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.color": GRID_COLOR,
    "grid.linewidth": 0.8,
    "axes.axisbelow": True
})

ranked_models = results_df.sort_values("RMSE")["Model"].tolist()
BASELINE = "Baseline (median EUI)"
scatter_models = [name for name in ranked_models if name != BASELINE]


# 1. Actual vs predicted EUI on the test data, one panel per model
fig, axes = plt.subplots(2, 2, figsize=(11, 10), sharex=True, sharey=True)
min_val = min(y_test.min(), min(p.min() for p in all_predictions.values()))
max_val = max(y_test.max(), max(p.max() for p in all_predictions.values()))

for ax, name in zip(axes.flat, scatter_models):
    r2 = results_df.loc[results_df["Model"] == name, "R2"].iloc[0]
    ax.scatter(
        y_test, all_predictions[name],
        s=14, alpha=0.4, color=SERIES_COLOR, edgecolors="none"
    )
    ax.plot(
        [min_val, max_val], [min_val, max_val],
        linestyle="--", linewidth=1.5, color=SECONDARY_TEXT
    )
    ax.set_title(f"{name}  (R² = {r2:.2f})", loc="left", fontsize=12)

for ax in axes[1]:
    ax.set_xlabel("Actual EUI (kBtu/ft²)")
for ax in axes[:, 0]:
    ax.set_ylabel("Predicted EUI (kBtu/ft²)")

fig.suptitle(
    "Actual vs predicted EUI on the test data\n"
    "Points on the dashed line are perfect predictions",
    x=0.06, ha="left", fontsize=14
)
fig.tight_layout()
fig.savefig("chart_actual_vs_predicted.png", dpi=150)


# 2. Model accuracy: R² and RMSE side by side (separate scales, no dual axis)
fig, (ax_r2, ax_rmse) = plt.subplots(1, 2, figsize=(12, 4.5))
ordered = results_df.set_index("Model").loc[ranked_models[::-1]]
bar_colors = [MUTED_COLOR if name == BASELINE else SERIES_COLOR for name in ordered.index]

ax_r2.barh(ordered.index, ordered["R2"], color=bar_colors, height=0.6)
ax_r2.set_title("R² (higher is better)", loc="left")
ax_r2.set_xlim(0, 1)
for i, value in enumerate(ordered["R2"]):
    ax_r2.text(max(value, 0) + 0.02, i, f"{value:.2f}", va="center", color=TEXT_COLOR)

ax_rmse.barh(ordered.index, ordered["RMSE"], color=bar_colors, height=0.6)
ax_rmse.set_title("RMSE in kBtu/ft² (lower is better)", loc="left")
ax_rmse.set_yticklabels([])
for i, value in enumerate(ordered["RMSE"]):
    ax_rmse.text(value + 0.5, i, f"{value:.1f}", va="center", color=TEXT_COLOR)

for ax in (ax_r2, ax_rmse):
    ax.grid(axis="y", visible=False)

fig.suptitle("Model accuracy on the test data", x=0.02, ha="left", fontsize=14)
fig.tight_layout()
fig.savefig("chart_model_accuracy.png", dpi=150)


# 3. Linear regression p-values per feature
p_chart = p_values_df[p_values_df["Feature"] != "Intercept"].copy()
p_chart["Label"] = (
    p_chart["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
    .str.replace("Multifamily Housing - ", "", regex=False)
    .str.replace("Largest Property Use Type - ", "", regex=False)
    .str.replace("Resident Population Type_", "Population: ", regex=False)
)
p_chart["Significance"] = -np.log10(p_chart["p-value"])
p_chart = p_chart.sort_values("Significance")

significance_line = -np.log10(0.05)
colors = [
    SERIES_COLOR if value > significance_line else MUTED_COLOR
    for value in p_chart["Significance"]
]

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(p_chart["Label"], p_chart["Significance"], color=colors, height=0.6)
ax.axvline(significance_line, linestyle="--", linewidth=1.5, color=SECONDARY_TEXT)
ax.text(
    significance_line + 0.3, len(p_chart) - 0.4, "p = 0.05 cut-off",
    color=SECONDARY_TEXT, fontsize=9
)
for i, p_value in enumerate(p_chart["p-value"]):
    # Keep labels clear of the cut-off line for short bars
    label_x = max(p_chart["Significance"].iloc[i], significance_line) + 0.3
    ax.text(
        label_x, i, f"p = {p_value:.2g}",
        va="center", color=TEXT_COLOR, fontsize=9
    )
ax.set_xlabel("−log10(p-value)   (further right = more significant)")
ax.grid(axis="y", visible=False)
ax.set_xlim(0, p_chart["Significance"].max() * 1.15)
ax.set_title(
    "Which features significantly affect EUI? (linear regression)\n"
    "Blue = significant at p < 0.05, grey = not significant",
    loc="left", fontsize=13
)
fig.tight_layout()
fig.savefig("chart_p_values.png", dpi=150)


# 4. Random Forest feature importance
forest_pipeline = fitted_pipelines["Random Forest"]
importance_df = pd.DataFrame({
    "Feature": forest_pipeline.named_steps["preprocessor"].get_feature_names_out(),
    "Importance": forest_pipeline.named_steps["regressor"].feature_importances_
})
importance_df["Label"] = (
    importance_df["Feature"]
    .str.replace("numeric__", "", regex=False)
    .str.replace("categorical__", "", regex=False)
    .str.replace("Multifamily Housing - ", "", regex=False)
    .str.replace("Largest Property Use Type - ", "", regex=False)
    .str.replace("Resident Population Type_", "Population: ", regex=False)
)
importance_df = importance_df.sort_values("Importance")

fig, ax = plt.subplots(figsize=(11, 6))
ax.barh(importance_df["Label"], importance_df["Importance"], color=SERIES_COLOR, height=0.6)
for i, value in enumerate(importance_df["Importance"]):
    ax.text(value + 0.005, i, f"{value:.1%}", va="center", color=TEXT_COLOR, fontsize=9)
ax.set_xlabel("Share of the model's decisions")
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.grid(axis="y", visible=False)
ax.set_xlim(0, importance_df["Importance"].max() * 1.15)
ax.set_title("What the Random Forest relies on most", loc="left", fontsize=13)
fig.tight_layout()
fig.savefig("chart_feature_importance.png", dpi=150)

plt.show()
