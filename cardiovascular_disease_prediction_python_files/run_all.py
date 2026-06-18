"""Run the cardiovascular disease prediction workflow end to end.
"""

from pathlib import Path

ROOT = Path(__file__).resolve().parent

FILES = [
    "cardio_01_setup.py",
    "cardio_02_load_data.py",
    "cardio_03_initial_checks.py",
    "cardio_04_cleaning.py",
    "cardio_05_cleaned_summary.py",
    "cardio_06_correlation_analysis.py",
    "cardio_07_train_test_preprocessing.py",
    "cardio_08_model_selection.py",
    "cardio_09_final_evaluation.py",
    "cardio_10_roc_curves.py",
    "cardio_11_confusion_matrix.py",
    "cardio_12_threshold_metrics.py",
    "cardio_13_logistic_coefficients.py",
    "cardio_14_permutation_importance.py",
    "cardio_15_calibration_check.py",
]

namespace = {"__name__": "__main__"}
for filename in FILES:
    path = ROOT / filename
    print(f"\n===== Running {filename} =====")
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), namespace)
