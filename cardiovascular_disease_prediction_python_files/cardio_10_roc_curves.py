# Generated from Code_cardiovascular_disease_prediction.ipynb
# One file corresponds to one non-empty code cell from the notebook.
# Run the files in numeric order, or use run_all.py.

plt.figure(figsize=(7.2, 5.5))

for name, (fpr, tpr, _) in roc_data.items():
    auc = results_df.loc[name, "Test AUC"]
    plt.plot(fpr, tpr, linewidth=2, label=f"{name} AUC = {auc:.3f}")

plt.plot([0, 1], [0, 1], "--", linewidth=1.4, label="Random classifier")
plt.xlabel("False positive rate")
plt.ylabel("True positive rate")
plt.title("ROC curves on the test set")
plt.legend(loc="lower right")
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "figure_2_roc_curves.pdf", bbox_inches="tight")
plt.show()
