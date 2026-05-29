# Generated from Code_cardiovascular_disease_prediction.ipynb
# One file corresponds to one non-empty code cell from the notebook.
# Run the files in numeric order, or use run_all.py.

rf_pipe = models["Random forest"].best_estimator_

perm = permutation_importance(
    rf_pipe,
    X_test,
    y_test,
    scoring="roc_auc",
    n_repeats=5,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

perm_df = pd.DataFrame({
    "variable": X_test.columns,
    "importance_mean_auc": perm.importances_mean,
    "importance_std_auc": perm.importances_std,
}).sort_values("importance_mean_auc", ascending=True)

display(perm_df.sort_values("importance_mean_auc", ascending=False).round(4))

plt.figure(figsize=(6.6, 4.8))
plt.barh(
    perm_df["variable"],
    perm_df["importance_mean_auc"],
    xerr=perm_df["importance_std_auc"],
    capsize=3,
)
plt.xlabel("Mean decrease in AUC")
plt.title("Permutation importance — Random forest")
plt.grid(axis="x", alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "figure_5_permutation_importance.pdf", bbox_inches="tight")
plt.show()
