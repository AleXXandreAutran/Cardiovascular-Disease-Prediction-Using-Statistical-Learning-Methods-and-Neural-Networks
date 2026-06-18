# Run the files in numeric order, or use run_all.py.

rf_proba = models["Random forest"].best_estimator_.predict_proba(X_test)[:, 1]
ridge_proba = models["Ridge"].best_estimator_.predict_proba(X_test)[:, 1]

rf_brier = brier_score_loss(y_test, rf_proba)
ridge_brier = brier_score_loss(y_test, ridge_proba)

print(f"Random forest Brier score:             {rf_brier:.4f}")
print(f"Ridge logistic regression Brier score: {ridge_brier:.4f}")
