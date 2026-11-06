import pandas as pd
import numpy as np
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Font, PatternFill, Alignment
from openpyxl.utils import get_column_letter
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import GradientBoostingRegressor

from data_prep import (
    test_df,
    energy_col,
    area_col,
    X_train,
    y_train,
    X_test,
    numeric_features,
    categorical_features
)

OUTPUT_FILE = "../test_data_with_predictions.xlsx"

# Row positions are used to match each building to its prediction
test_df = test_df.reset_index(drop=True)

preprocessor = ColumnTransformer([
    ("numeric", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric_features),
    ("categorical", Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]), categorical_features)
])

model = Pipeline([
    ("preprocessor", preprocessor),
    # Best model on the honest train/test split (see multi-predict.py)
    ("regressor", GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42
    ))
])

model.fit(X_train, y_train)
predictions = model.predict(X_test)
median_training_eui = y_train.median()


# Build the workbook
FONT = "Arial"
header_font = Font(name=FONT, bold=True, color="FFFFFF")
header_fill = PatternFill("solid", start_color="2A78D6")
added_fill = PatternFill("solid", start_color="1F4E8C")
body_font = Font(name=FONT)
input_font = Font(name=FONT, color="0000FF")
bold_font = Font(name=FONT, bold=True)

wb = Workbook()
sheet = wb.active
sheet.title = "Predictions"

# Original test columns, minus the EUI the script calculated (re-added as a formula)
original_columns = [col for col in test_df.columns if col != "EUI"]
added_columns = [
    "Actual EUI (kBtu/ft²)",
    "Predicted EUI - Gradient Boosting (kBtu/ft²)",
    "Difference (Predicted - Actual)",
    "% Difference",
    "Baseline EUI - Median (kBtu/ft²)",
    "Baseline Difference (Baseline - Actual)"
]
headers = original_columns + added_columns

for col_index, header in enumerate(headers, start=1):
    cell = sheet.cell(row=1, column=col_index, value=header)
    cell.font = header_font
    cell.fill = added_fill if header in added_columns else header_fill
    cell.alignment = Alignment(wrap_text=True, vertical="top")

column_letter = {header: get_column_letter(i) for i, header in enumerate(headers, start=1)}
energy = column_letter[energy_col]
area = column_letter[area_col]
actual = column_letter["Actual EUI (kBtu/ft²)"]
predicted = column_letter["Predicted EUI - Gradient Boosting (kBtu/ft²)"]
difference = column_letter["Difference (Predicted - Actual)"]
baseline = column_letter["Baseline EUI - Median (kBtu/ft²)"]
baseline_difference = column_letter["Baseline Difference (Baseline - Actual)"]

for row_index, row in test_df.iterrows():
    excel_row = row_index + 2

    for col_index, col in enumerate(original_columns, start=1):
        value = row[col]
        if pd.isna(value):
            value = None
        elif isinstance(value, np.generic):
            value = value.item()
        sheet.cell(row=excel_row, column=col_index, value=value).font = body_font

    added_values = [
        f"={energy}{excel_row}/{area}{excel_row}",
        round(float(predictions[row_index]), 4),
        f"={predicted}{excel_row}-{actual}{excel_row}",
        f"=IF({actual}{excel_row}=0,\"\",{difference}{excel_row}/{actual}{excel_row})",
        "=Summary!$B$4",
        f"={baseline}{excel_row}-{actual}{excel_row}"
    ]
    for offset, value in enumerate(added_values):
        cell = sheet.cell(
            row=excel_row, column=len(original_columns) + 1 + offset, value=value
        )
        cell.font = body_font
        cell.number_format = "0.0%" if offset == 3 else "0.00"

last_row = len(test_df) + 1
sheet.freeze_panes = "B2"
sheet.auto_filter.ref = f"A1:{get_column_letter(len(headers))}{last_row}"
sheet.row_dimensions[1].height = 60
for header, letter in column_letter.items():
    sheet.column_dimensions[letter].width = 18 if header in added_columns else 14


# Summary sheet
summary = wb.create_sheet("Summary", 0)
summary.column_dimensions["A"].width = 44
summary.column_dimensions["B"].width = 16
summary.column_dimensions["C"].width = 16

summary["A1"] = "Predicted vs actual EUI on the test data"
summary["A1"].font = Font(name=FONT, bold=True, size=14)
summary["A2"] = f"{len(test_df)} buildings, see the Predictions tab for each one"
summary["A2"].font = body_font

summary["A4"] = "Baseline: median EUI of the training data"
summary["B4"] = round(float(median_training_eui), 4)
summary["B4"].font = input_font
summary["B4"].number_format = "0.00"
summary["B4"].comment = Comment(
    "Calculated by export_predictions.py: median EUI of the training dataset. "
    "Change it here to test a different baseline.", "export_predictions.py"
)
summary["A4"].font = body_font

summary["B6"] = "Gradient Boosting"
summary["C6"] = "Baseline"
summary["A6"] = "Metric"
for cell in (summary["A6"], summary["B6"], summary["C6"]):
    cell.font = header_font
    cell.fill = header_fill

data = "Predictions!"
metrics = [
    (
        "MAE: average size of the error (kBtu/ft²)",
        f"=SUMPRODUCT(ABS({data}{difference}2:{difference}{last_row}))/COUNT({data}{difference}2:{difference}{last_row})",
        f"=SUMPRODUCT(ABS({data}{baseline_difference}2:{baseline_difference}{last_row}))/COUNT({data}{baseline_difference}2:{baseline_difference}{last_row})"
    ),
    (
        "RMSE: penalises big misses more (kBtu/ft²)",
        f"=SQRT(SUMSQ({data}{difference}2:{difference}{last_row})/COUNT({data}{difference}2:{difference}{last_row}))",
        f"=SQRT(SUMSQ({data}{baseline_difference}2:{baseline_difference}{last_row})/COUNT({data}{baseline_difference}2:{baseline_difference}{last_row}))"
    ),
    (
        "Average difference (+ = over-predicts on average)",
        f"=AVERAGE({data}{difference}2:{difference}{last_row})",
        f"=AVERAGE({data}{baseline_difference}2:{baseline_difference}{last_row})"
    )
]
for i, (label, model_formula, baseline_formula) in enumerate(metrics, start=7):
    summary[f"A{i}"] = label
    summary[f"B{i}"] = model_formula
    summary[f"C{i}"] = baseline_formula
    summary[f"A{i}"].font = body_font
    for col in ("B", "C"):
        summary[f"{col}{i}"].font = bold_font
        summary[f"{col}{i}"].number_format = "0.00"

summary["A11"] = "Note"
summary["A11"].font = bold_font
summary["A12"] = (
    "The model was trained on a separate file of buildings "
    "(multifamily_building_energy_consumption_training_dataset.csv) and never saw these ones, "
    "so these errors show how it does on new buildings."
)
summary["A12"].font = body_font
summary["A12"].alignment = Alignment(wrap_text=True, vertical="top")
summary.merge_cells("A12:C12")
summary.row_dimensions[12].height = 45

wb.save(OUTPUT_FILE)
print(f"Saved {len(test_df)} rows to {OUTPUT_FILE}")
