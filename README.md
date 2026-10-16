# Cardiovascular Disease Prediction Using Statistical Learning Methods and Neural Networks

This project compares five models for classifying the cardiovascular disease label in a public dataset: Ridge and Lasso logistic regression, nearest neighbors, random forest, and a small neural network.

Each row is one patient record. There is no follow-up period, so these models describe the recorded disease label, not the chance of a future cardiovascular event. This is a learning project, not a medical tool.

## Run it

Tested with Python 3.12.14 and the versions in `requirements.txt`.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment with `.venv\Scripts\activate`.

Download `cardio_train.csv` from the [dataset page](https://www.kaggle.com/datasets/sulianova/cardiovascular-disease-dataset) and place it in `data/`. Then run:

```bash
python cardiovascular.py --jobs 2
```

For a shorter first run:

```bash
python cardiovascular.py --quick --jobs 2
```

The quick option uses fewer settings and trees. It is a check that the workflow runs, not the full model search. You can also give another input or output path:

```bash
python cardiovascular.py --data /path/to/cardio_train.csv --output /path/to/results --jobs 2
```

Relative input and output paths are resolved from the repository folder, even if you launch the script elsewhere. Running it again replaces files with the same names in the output folder.

## How it works

The script converts age to years and computes BMI. It removes the identifier. It keeps age, height, BMI, both blood pressures, gender, cholesterol, glucose, smoking, alcohol use, and activity as predictors. Weight is used to compute BMI and is not a separate predictor. The categorical codes keep their original numeric encoding; they are not one-hot encoded.

The cleaning rules keep the original project's assumptions. Negative blood pressures become positive. Pressures of at least 1,000 are divided by 10 once. Systolic values from 11 to 25 and diastolic values from 4 to 15 are multiplied by 10. A reversed pair is swapped when the proposed pair falls in the stated ranges. These are assumed recording errors, not verified corrections.

Rows must then have age 18–100 years, height 130–210 cm, weight 40–200 kg, BMI 15–50, systolic pressure 60–250, diastolic pressure 40–150, and systolic pressure at least as high as diastolic pressure. Endpoints are included. Rows with missing values or invalid category codes are dropped. The cleaning counts are saved.

After cleaning, an 80/20 stratified split separates training and test data. Five-fold cross-validation on training data tunes each model using ROC AUC. Scaling is fitted inside each fold. Only the five continuous predictors are standardized; the category codes pass through unchanged. The model family with the highest mean validation AUC is chosen before the test set is used. The final decision threshold stays at 0.5.

Threshold plots use out-of-fold predictions from the training set for the selected settings. They help show the precision–recall trade-off. Because those settings were selected using the same training data, these plots are exploratory; they are not an independent performance estimate and do not change the final threshold.

The selected model is evaluated once on the held-out test set. Its ROC curve, precision–recall curve, confusion matrix, and permutation importance describe that evaluation. Test results are not used to select a model, change the threshold, or remove features. Permutation importance describes the fitted model and does not establish cause and effect. Correlated predictors can share importance.

## Saved files

`results/` contains:

- `run_summary.json`: settings, input file SHA-256 checksum, cleaning counts, selected model, and test metrics.
- `cv_results.csv`: the validation scores and chosen settings for all five models.
- `search_*.csv`: each model's full validation search.
- `test_metrics.csv` and `test_predictions.csv`: the selected model's test results at threshold 0.5.
- `training_thresholds.csv`: the exploratory threshold comparison on training predictions.
- `logistic_coefficients.csv`: coefficients for the two logistic models. Continuous predictors are standardized; the other coefficients use the original category codes.
- `permutation_importance.csv`: changes in test AUC after shuffling each predictor.
- `figures/`: PNG plots, including a reliability diagram.

The Brier score measures squared probability error; lower values are better. It reflects both calibration and the ability to separate the classes, so it is not a calibration test on its own. A constant prediction based on training prevalence is included as a reference. The reliability diagram gives a separate view of calibration.

The split and random seeds are fixed at 42. Exact values can still vary slightly across machines. The data come from one source, and the pressure corrections are assumptions. Results need independent validation before any practical use.
