import argparse
import hashlib
import json
from pathlib import Path
import platform

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from joblib import parallel_config
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    train_test_split,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits


ROOT = Path(__file__).resolve().parent
SEED = 42
NUMERIC = ["age", "height", "bmi", "ap_hi", "ap_lo"]
CATEGORIES = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
FEATURES = NUMERIC + CATEGORIES
REQUIRED = ["age", "height", "weight", "ap_hi", "ap_lo"] + CATEGORIES + ["cardio"]
CODES = {
    "gender": [1, 2],
    "cholesterol": [1, 2, 3],
    "gluc": [1, 2, 3],
    "smoke": [0, 1],
    "alco": [0, 1],
    "active": [0, 1],
    "cardio": [0, 1],
}


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_data(path):
    if not path.is_file():
        raise ValueError(f"Data file not found: {path}. See data/README.md.")
    frame = pd.read_csv(path, sep=None, engine="python")
    frame.columns = frame.columns.str.strip()
    missing = sorted(set(REQUIRED) - set(frame.columns))
    if missing:
        raise ValueError(f"Missing columns: {', '.join(missing)}")
    return frame


def clean_data(raw):
    frame = raw[REQUIRED].copy()
    for column in REQUIRED:
        if not pd.api.types.is_numeric_dtype(frame[column]):
            frame[column] = frame[column].astype("string").str.replace(",", ".", regex=False)
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.replace([np.inf, -np.inf], np.nan)
    frame["age"] = frame["age"] / 365.25
    for column in ["ap_hi", "ap_lo"]:
        frame[column] = frame[column].astype(float).abs()
        frame.loc[frame[column] >= 1000, column] /= 10
    frame.loc[frame["ap_hi"].between(11, 25), "ap_hi"] *= 10
    frame.loc[frame["ap_lo"].between(4, 15), "ap_lo"] *= 10
    swap = (
        (frame["ap_lo"] > frame["ap_hi"])
        & frame["ap_lo"].between(60, 250)
        & frame["ap_hi"].between(40, 150)
    )
    frame.loc[swap, ["ap_hi", "ap_lo"]] = frame.loc[
        swap, ["ap_lo", "ap_hi"]
    ].to_numpy()
    frame["bmi"] = frame["weight"] / (frame["height"] / 100) ** 2
    keep = (
        frame["age"].between(18, 100)
        & frame["height"].between(130, 210)
        & frame["weight"].between(40, 200)
        & frame["bmi"].between(15, 50)
        & frame["ap_hi"].between(60, 250)
        & frame["ap_lo"].between(40, 150)
        & (frame["ap_hi"] >= frame["ap_lo"])
        & frame.notna().all(axis=1)
    )
    for column, valid_codes in CODES.items():
        keep &= frame[column].isin(valid_codes)
    frame = frame.loc[keep].copy()
    for column in CODES:
        frame[column] = frame[column].astype(int)
    frame.index.name = "source_row"
    counts = {"raw_rows": len(raw), "clean_rows": len(frame), "removed_rows": len(raw) - len(frame)}
    return frame, counts


def model_specs(quick):
    regularization = [0.01, 1] if quick else [0.001, 0.01, 0.1, 1, 10, 100]
    return {
        "Ridge": (
            LogisticRegression(l1_ratio=0.0, solver="lbfgs", max_iter=5000, random_state=SEED),
            {"model__C": regularization},
        ),
        "Lasso": (
            LogisticRegression(l1_ratio=1.0, solver="liblinear", max_iter=5000, random_state=SEED),
            {"model__C": regularization},
        ),
        "KNN": (
            KNeighborsClassifier(n_jobs=1),
            {"model__n_neighbors": [50] if quick else [20, 50, 100], "model__weights": ["uniform"]},
        ),
        "Random forest": (
            RandomForestClassifier(random_state=SEED, n_jobs=1),
            {
                "model__n_estimators": [100] if quick else [100, 300],
                "model__max_depth": [8] if quick else [8, 12, None],
                "model__min_samples_leaf": [5] if quick else [5, 10],
            },
        ),
        "MLP": (
            MLPClassifier(activation="relu", solver="adam", early_stopping=True, max_iter=300, random_state=SEED),
            {
                "model__hidden_layer_sizes": [(50,)] if quick else [(50,), (100,), (50, 25)],
                "model__alpha": [0.001] if quick else [0.0001, 0.001],
                "model__learning_rate_init": [0.001],
            },
        ),
    }


def score_predictions(y, probabilities, threshold=0.5):
    predicted = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predicted)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }


def save_figure(fig, folder, name):
    fig.tight_layout()
    fig.savefig(folder / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_plots(frame, y_test, probabilities, best_name, thresholds, importance, folder):
    corr = frame[FEATURES + ["cardio"]].corr()
    fig, ax = plt.subplots(figsize=(9, 8))
    color = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=90)
    ax.set_yticks(range(len(corr)), corr.columns)
    ax.set_title("Correlations after cleaning")
    fig.colorbar(color, ax=ax, label="Pearson correlation")
    save_figure(fig, folder, "correlations.png")

    fpr, tpr, _ = roc_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"{best_name}: AUC = {roc_auc_score(y_test, probabilities):.3f}")
    ax.plot([0, 1], [0, 1], "--", color="gray")
    ax.set(xlabel="False positive rate", ylabel="True positive rate", title="Held-out test ROC curve")
    ax.legend(loc="lower right")
    save_figure(fig, folder, "test_roc.png")

    precision, recall, _ = precision_recall_curve(y_test, probabilities)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, label=f"Average precision = {average_precision_score(y_test, probabilities):.3f}")
    ax.axhline(y_test.mean(), linestyle="--", color="gray", label="Test prevalence")
    ax.set(xlabel="Recall", ylabel="Precision", title=f"Held-out test: {best_name}")
    ax.legend()
    save_figure(fig, folder, "test_precision_recall.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test, (probabilities >= 0.5).astype(int), labels=[0, 1],
        display_labels=["Absent", "Present"], colorbar=False, cmap="Blues", ax=ax,
    )
    ax.set_title(f"Held-out test: {best_name}, threshold 0.5")
    save_figure(fig, folder, "test_confusion_matrix.png")

    fig, ax = plt.subplots(figsize=(6, 5))
    for metric in ["precision", "recall", "f1"]:
        ax.plot(thresholds["threshold"], thresholds[metric], marker="o", label=metric.title())
    ax.set(xlabel="Threshold", ylabel="Score", ylim=(0, 1), title="Training out-of-fold predictions (exploratory)")
    ax.legend()
    save_figure(fig, folder, "training_thresholds.png")

    observed, predicted = calibration_curve(y_test, probabilities, n_bins=10, strategy="quantile")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(predicted, observed, marker="o", label=best_name)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="Ideal calibration")
    ax.set(xlabel="Mean predicted probability", ylabel="Observed positive fraction", title="Held-out test reliability diagram", xlim=(0, 1), ylim=(0, 1))
    ax.legend()
    save_figure(fig, folder, "test_calibration.png")

    ordered = importance.sort_values("mean_auc_decrease")
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.barh(ordered["feature"], ordered["mean_auc_decrease"], xerr=ordered["std_auc_decrease"], capsize=3)
    ax.set(xlabel="Mean decrease in test AUC", title=f"Permutation importance: {best_name}")
    save_figure(fig, folder, "test_permutation_importance.png")


def run(args):
    data_path = resolve_path(args.data)
    output = resolve_path(args.output)
    frame, counts = clean_data(load_data(data_path))
    data_sha256 = hashlib.sha256(data_path.read_bytes()).hexdigest()
    if frame["cardio"].value_counts().reindex([0, 1], fill_value=0).min() < 10:
        raise ValueError("At least 10 cleaned rows in each class are needed for the stratified split and five folds.")
    X_train, X_test, y_train, y_test = train_test_split(
        frame[FEATURES], frame["cardio"], test_size=0.2, stratify=frame["cardio"], random_state=SEED,
    )
    output.mkdir(parents=True, exist_ok=True)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    largest_k = max(model_specs(args.quick)["KNN"][1]["model__n_neighbors"])
    smallest_fold = min(len(train) for train, _ in cv.split(X_train, y_train))
    if smallest_fold < largest_k:
        raise ValueError(f"The smallest training fold has {smallest_fold} rows; KNN needs at least {largest_k}.")
    searches = {}
    rows = []
    for name, (estimator, settings) in model_specs(args.quick).items():
        print(f"Training {name}...", flush=True)
        preprocess = ColumnTransformer([
            ("numeric", StandardScaler(), NUMERIC),
            ("categories", "passthrough", CATEGORIES),
        ])
        pipeline = Pipeline([("preprocess", preprocess), ("model", estimator)])
        search = GridSearchCV(pipeline, settings, scoring="roc_auc", cv=cv, n_jobs=args.jobs, refit=True, error_score="raise")
        search.fit(X_train, y_train)
        searches[name] = search
        rows.append({
            "model": name,
            "mean_cv_auc": float(search.best_score_),
            "std_cv_auc": float(search.cv_results_["std_test_score"][search.best_index_]),
            "parameters": json.dumps(search.best_params_),
        })
        slug = name.lower().replace(" ", "_")
        pd.DataFrame(search.cv_results_).to_csv(output / f"search_{slug}.csv", index=False)
        print(f"  Mean validation AUC: {search.best_score_:.4f}", flush=True)
    validation = pd.DataFrame(rows).sort_values("mean_cv_auc", ascending=False, kind="stable")
    validation.to_csv(output / "cv_results.csv", index=False)
    best_name = validation.iloc[0]["model"]
    selected = searches[best_name].best_estimator_
    print(f"Selected by validation AUC: {best_name}. Test threshold is fixed at 0.5.", flush=True)

    training_probabilities = cross_val_predict(
        selected, X_train, y_train, cv=cv, method="predict_proba", n_jobs=args.jobs,
    )[:, 1]
    thresholds = pd.DataFrame([
        score_predictions(y_train, training_probabilities, threshold)
        for threshold in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
    ])
    thresholds.to_csv(output / "training_thresholds.csv", index=False)

    probabilities = selected.predict_proba(X_test)[:, 1]
    metrics = score_predictions(y_test, probabilities)
    metrics.update({
        "model": best_name,
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "average_precision": float(average_precision_score(y_test, probabilities)),
        "brier_score": float(brier_score_loss(y_test, probabilities)),
        "prevalence_baseline_brier": float(brier_score_loss(y_test, np.full(len(y_test), y_train.mean()))),
    })
    pd.DataFrame([metrics]).to_csv(output / "test_metrics.csv", index=False)
    predictions = pd.DataFrame({
        "source_row": X_test.index, "observed": y_test.to_numpy(),
        "probability": probabilities, "predicted_at_0_5": (probabilities >= 0.5).astype(int),
    })
    predictions.to_csv(output / "test_predictions.csv", index=False)
    coefficient_rows = []
    for name in ["Ridge", "Lasso"]:
        coefficients = searches[name].best_estimator_.named_steps["model"].coef_.ravel()
        coefficient_rows.extend({"model": name, "feature": feature, "coefficient": float(coefficient)} for feature, coefficient in zip(FEATURES, coefficients))
    pd.DataFrame(coefficient_rows).to_csv(output / "logistic_coefficients.csv", index=False)
    permutation = permutation_importance(
        selected, X_test, y_test, scoring="roc_auc", n_repeats=2 if args.quick else 5, random_state=SEED, n_jobs=args.jobs,
    )
    importance = pd.DataFrame({
        "feature": FEATURES, "mean_auc_decrease": permutation.importances_mean,
        "std_auc_decrease": permutation.importances_std,
    }).sort_values("mean_auc_decrease", ascending=False)
    importance.to_csv(output / "permutation_importance.csv", index=False)
    draw_plots(frame, y_test, probabilities, best_name, thresholds, importance, figures)
    summary = {
        "data_file": str(data_path), "input_sha256": data_sha256, "seed": SEED, "quick": args.quick,
        "cleaning": counts, "training_rows": len(X_train), "test_rows": len(X_test),
        "training_prevalence": float(y_train.mean()), "test_prevalence": float(y_test.mean()),
        "selection_rule": "Highest mean five-fold training CV ROC AUC; fixed threshold 0.5",
        "selected_model": best_name, "selected_parameters": searches[best_name].best_params_,
        "test_metrics": metrics,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
    }
    (output / "run_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Cleaned rows: {len(frame):,}; removed: {counts['removed_rows']:,}")
    print(f"Held-out test AUC: {metrics['roc_auc']:.4f}; F1 at 0.5: {metrics['f1']:.4f}")
    print(f"Results saved to {output}")


def main():
    parser = argparse.ArgumentParser(description="Compare models for the recorded cardiovascular disease label.")
    parser.add_argument("--data", default="data/cardio_train.csv", help="CSV path, relative to this script's folder or absolute")
    parser.add_argument("--output", default="results", help="Output folder, relative to this script's folder or absolute")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel workers; use -1 for all available cores")
    parser.add_argument("--quick", action="store_true", help="Use a smaller grid for a first run")
    args = parser.parse_args()
    if args.jobs == 0 or args.jobs < -1:
        parser.error("--jobs must be a positive integer or -1")
    try:
        with threadpool_limits(limits=1), parallel_config(backend="loky", inner_max_num_threads=1):
            run(args)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Error: {error}\n")


if __name__ == "__main__":
    main()
