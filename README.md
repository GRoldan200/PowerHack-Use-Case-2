# PowerHack-Use-Case-2
PowerHack Use Case 2 files
THIS WILL BE USED FOR THE PRESENTATION

# Building Energy Prediction

Predicts the energy use intensity (EUI, energy use per square foot) of NYC multifamily buildings from their characteristics, and compares several models against a simple baseline.

**Data source:** NYC multifamily building characteristics and annual usage data, prepared and cleaned by NYPA. Field names follow ENERGY STAR Portfolio Manager.

## Results

The models are trained on the training file and scored on the separate testing file, which has 2,495 buildings that the models never saw. The two files share no buildings.

| Model | MAE (kBtu/ft²) | RMSE (kBtu/ft²) | R² |
|---|---|---|---|
| **Gradient Boosting** | **21.69** | **32.78** | **0.20** |
| Ridge Regression | 22.54 | 33.22 | 0.18 |
| Linear Regression | 22.54 | 33.23 | 0.18 |
| Random Forest | 21.60 | 33.79 | 0.15 |
| Baseline (median EUI of 75.64 for every building) | 25.14 | 36.66 | 0.00 |

- **Against the baseline:** Gradient Boosting's MAE is 13.7% lower and its RMSE 10.6% lower.
- **Extra features:** adding density, laundry and high-rise/mid-rise features raised R² from 0.11 to 0.20.
- **MAPE:** it isn't reported. A few test buildings have an EUI below 3, so their percentage errors run into the thousands and would make MAPE meaningless.

## Key findings

1. **The best model beats the baseline, but only by a modest margin.** Gradient Boosting cuts the average error from 25.1 to 21.7 kBtu/ft². Adding building density (units and bedrooms per 1,000 ft²) roughly doubled the accuracy.
2. **Building age and density matter most.** Year built is the strongest predictor: newer buildings use less energy per square foot (p ≈ 1e-62). Units per 1,000 ft² comes second. Laundry hookups, high-rise status and resident population type barely change the predictions.
3. **The data lacks the real drivers of energy use and has quality problems.** An R² of 0.20 leaves 80% of the variation unexplained. Heating fuel, HVAC type, insulation and climate aren't in the dataset. Some values were also recorded as `0` when the data wasn't available.

## What cannot be concluded

- **Building age:** newer buildings had lower EUI, but that doesn't prove age is the cause. Newer buildings may also have better insulation, windows or heating systems, which aren't in the data. Year built is the strongest *predictor*, not a proven *cause*.
- **Unit density:** buildings with more units per 1,000 ft² had higher EUI (p ≈ 6e-39). That doesn't prove crowding raises energy use. Density and EUI are both divided by floor area, so part of the link may come from that shared divisor.
- **Student housing:** "Dedicated Student" buildings had EUI about 37 kBtu/ft² lower, but there are only 11 such buildings in the training data. That's too few to conclude anything.
- **Laundry hookups:** they barely affected predictions, but that doesn't show laundry is unimportant. In the testing file, 86.5% of laundry values are `0`, with real zeros and missing data mixed together.
- **Best model:** the models' scores are close (R² 0.15–0.20) and come from a single test file, so Gradient Boosting isn't proven to be the best model.
- **Individual buildings:** an average error of 21.7 kBtu/ft² on a typical EUI of about 75 is roughly a 30% miss. The model isn't reliable for judging one building.
- **Buildings above their prediction:** using more energy than predicted isn't proof that a building is wasteful. It may have features the data doesn't capture.
- **The worst predictions:** the largest % errors were for buildings with an actual EUI of 0.19–2.90, where the model predicted 32–95. Those actual values are far below the typical 56–92 and are probably reporting errors, not model failures.
- **Other buildings:** the results come from NYC multifamily buildings and may not apply to offices, schools or buildings in other climates.

## Data and cleaning

| | Training file | Testing file |
|---|---|---|
| File | `multifamily_building_energy_consumption_training_dataset.csv` | `multifamily_building_energy_consumption_testing_dataset.csv` |
| Buildings in file | 9,976 | 2,495 |
| Buildings used | 9,886 | 2,495 (all kept) |

**Target:** EUI = `Weather Normalized Site Energy Use (kBtu)` ÷ `Largest Property Use Type - Gross Floor Area (ft^2)`.

**Cleaning:**

- **Implausible EUIs:** we removed 90 training buildings with an EUI below 5 or above 400, which are likely data errors. Test buildings are never removed, so the scores reflect real data.
- **Population type:** missing values (`Not Available` in training, `0` in testing) are grouped as `Unknown`.
- **Laundry hookups:** the testing file stores "not available" as `0`, so the training file's `Not Available` is recoded to `0` to match.
- **Out-of-range test values:** test values are limited to the range seen in training. For example, one test building lists 100,000 laundry hookups, which would otherwise send the linear models to an EUI of −2,541.
- **Header fix:** the testing file's floor-area header was saved as `(ft?)` and is renamed to `(ft^2)` when loaded.

**Features (14):** year built, occupancy, residential units, bedrooms, percent heated, percent cooled, laundry hookups (in units and in common areas), high-rise units, mid-rise units, the four density columns (bedrooms, units, high-rise units and mid-rise units per 1,000 ft²), and resident population type, one-hot encoded.

Energy use and EUI are never used as inputs. Floor area appears only through the density columns. It's known before energy is measured, so it doesn't leak the answer.

## Files

| File | Purpose |
|---|---|
| `data_prep.py` | Loads and cleans both data files; shared by the scripts below |
| `multi-predict.py` | Trains and compares all models, prints p-values, and saves the charts |
| `export_predictions.py` | Writes `../test_data_with_predictions.xlsx`: each test building with predicted EUI, difference and baseline |
| `main.py` | The original single linear-regression script. It hasn't been updated: it uses the old overlapping data and fails on the `(ft?)` header. |

**Charts** (created by `multi-predict.py`):

| Chart | Shows |
|---|---|
| `prediction/chart_actual_vs_predicted.png` | Actual vs predicted EUI for each model |
| `prediction/chart_model_accuracy.png` | R² and RMSE for each model against the baseline |
| `prediction/chart_p_values.png` | Which features are statistically significant in the linear regression |
| `prediction/chart_feature_importance.png` | Permutation importance for the best model, measured on the test data |

## How to run

From the `prediction` folder:

```bash
python -m venv venv
venv\Scripts\activate          # macOS/Linux: source venv/bin/activate
pip install -r requirements.txt

python multi-predict.py        # model comparison, p-values and charts
python export_predictions.py   # Excel file of per-building predictions
```

The data CSVs must be in the folder above `prediction`.
 
