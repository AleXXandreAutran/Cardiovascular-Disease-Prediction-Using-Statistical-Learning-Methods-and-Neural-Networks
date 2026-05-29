# Generated from Code_cardiovascular_disease_prediction.ipynb
# One file corresponds to one non-empty code cell from the notebook.
# Run the files in numeric order, or use run_all.py.

thresholds_to_check = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
threshold_rows = []

for threshold in thresholds_to_check:
    y_pred_t = (y_proba_best >= threshold).astype(int)
    cm_t = confusion_matrix(y_test, y_pred_t)
    threshold_rows.append({
        "threshold": threshold,
        "accuracy": accuracy_score(y_test, y_pred_t),
        "precision": precision_score(y_test, y_pred_t),
        "recall": recall_score(y_test, y_pred_t),
        "f1": f1_score(y_test, y_pred_t),
        "false_negatives": cm_t[1, 0],
        "false_positives": cm_t[0, 1],
    })

threshold_df = pd.DataFrame(threshold_rows)
display(threshold_df.round(3))

precision, recall, _ = precision_recall_curve(y_test, y_proba_best)
avg_precision = average_precision_score(y_test, y_proba_best)

plt.figure(figsize=(6.4, 4.8))
plt.plot(recall, precision, linewidth=2, label=f"Average precision = {avg_precision:.3f}")
plt.axhline(y_test.mean(), linestyle="--", linewidth=1.2, label=f"Baseline = {y_test.mean():.3f}")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title(f"Precision-recall curve — {best_name}")
plt.legend()
plt.grid(True, alpha=0.25)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "figure_4a_precision_recall.pdf", bbox_inches="tight")
plt.show()

fig, ax1 = plt.subplots(figsize=(6.8, 4.8))
ax1.plot(threshold_df["threshold"], threshold_df["precision"], marker="o", label="Precision")
ax1.plot(threshold_df["threshold"], threshold_df["recall"], marker="o", label="Recall")
ax1.plot(threshold_df["threshold"], threshold_df["f1"], marker="o", label="F1-score")
ax1.set_xlabel("Decision threshold")
ax1.set_ylabel("Score")
ax1.grid(True, alpha=0.25)

ax2 = ax1.twinx()
ax2.plot(threshold_df["threshold"], threshold_df["false_negatives"], marker="s", linestyle="--", label="False negatives")
ax2.plot(threshold_df["threshold"], threshold_df["false_positives"], marker="s", linestyle="--", label="False positives")
ax2.set_ylabel("Number of errors")

lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="best")
plt.title("Threshold-dependent clinical metrics")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "figure_4b_threshold_metrics.pdf", bbox_inches="tight")
plt.show()
