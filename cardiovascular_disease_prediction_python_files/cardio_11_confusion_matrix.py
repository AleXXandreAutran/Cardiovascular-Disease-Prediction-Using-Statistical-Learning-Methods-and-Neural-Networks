# Generated from Code_cardiovascular_disease_prediction.ipynb
# One file corresponds to one non-empty code cell from the notebook.
# Run the files in numeric order, or use run_all.py.

best_name = results_df.index[0]
best_model = models[best_name].best_estimator_
y_pred_best = best_model.predict(X_test)
y_proba_best = best_model.predict_proba(X_test)[:, 1]

cm = confusion_matrix(y_test, y_pred_best)
print(f"Best model according to test AUC: {best_name}")
print(cm)

ConfusionMatrixDisplay(cm, display_labels=["Predicted 0", "Predicted 1"]).plot(values_format="d", colorbar=False)
plt.title(f"Confusion matrix — {best_name}, threshold 0.5")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "figure_3_confusion_matrix.pdf", bbox_inches="tight")
plt.show()
