# Run the files in numeric order, or use run_all.py.

results = []
roc_data = {}

for name, grid in models.items():
    model = grid.best_estimator_
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Test AUC": roc_auc_score(y_test, y_proba),
        "CV AUC mean": grid.best_score_,
        "CV AUC std": grid.cv_results_["std_test_score"][grid.best_index_],
    })

    roc_data[name] = roc_curve(y_test, y_proba)

results_df = pd.DataFrame(results).set_index("Model").sort_values("Test AUC", ascending=False)
display(results_df.round(4))
