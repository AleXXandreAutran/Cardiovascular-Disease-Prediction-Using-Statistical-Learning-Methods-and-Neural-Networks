# Cardiovascular Disease Prediction — Python scripts

## How to run

Place the dataset here:

```text
data/cardiovascular_disease_dataset.csv
```

Then run the full workflow:

```bash
python run_all.py
```

The scripts save figures to the `figures/` directory.

## Files

- `cardio_01_setup.py`: imports, configuration, paths, plotting settings.
- `cardio_02_load_data.py`: dataset loading function and initial load.
- `cardio_03_initial_checks.py`: variable types, missing values, target balance, descriptive statistics.
- `cardio_04_cleaning.py`: deterministic cleaning and BMI feature creation.
- `cardio_05_cleaned_summary.py`: post-cleaning summaries.
- `cardio_06_correlation_analysis.py`: target correlations and correlation heatmap.
- `cardio_07_train_test_preprocessing.py`: feature selection, train-test split, preprocessing pipeline, cross-validation object.
- `cardio_08_model_selection.py`: cross-validated model selection for ridge, lasso, KNN, random forest, and MLP.
- `cardio_09_final_evaluation.py`: held-out test metrics and ROC data.
- `cardio_10_roc_curves.py`: ROC curve plot.
- `cardio_11_confusion_matrix.py`: best-model confusion matrix.
- `cardio_12_threshold_metrics.py`: threshold-dependent metrics and precision-recall plots.
- `cardio_13_logistic_coefficients.py`: standardized coefficients for ridge and lasso logistic regression.
- `cardio_14_permutation_importance.py`: random forest permutation importance.
- `cardio_15_calibration_check.py`: Brier score comparison.
- `run_all.py`: convenience runner that executes all exported cells in order.
