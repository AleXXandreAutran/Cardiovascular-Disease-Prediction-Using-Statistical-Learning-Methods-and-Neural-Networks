# Generated from Code_cardiovascular_disease_prediction.ipynb
# Run the files in numeric order, or use run_all.py.

corr_cols = [
    "age", "height", "weight", "bmi", "ap_hi", "ap_lo",
    "cholesterol", "gluc", "smoke", "alco", "active", "cardio",
]

corr = df_clean[corr_cols].apply(pd.to_numeric, errors="coerce").corr()

print("Correlations with the target:")
display(corr["cardio"].sort_values(ascending=False).to_frame("corr_cardio"))

plt.figure(figsize=(8.5, 7.5))
if sns is not None:
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        square=True,
        linewidths=0.5,
        cbar_kws={"label": "Pearson correlation"},
    )
else:
    plt.imshow(corr, vmin=-1, vmax=1)
    plt.colorbar(label="Pearson correlation")
    plt.xticks(range(len(corr.columns)), corr.columns, rotation=90)
    plt.yticks(range(len(corr.index)), corr.index)

plt.title("Correlation matrix after cleaning")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "figure_1_correlation_matrix.pdf", bbox_inches="tight")
plt.show()
