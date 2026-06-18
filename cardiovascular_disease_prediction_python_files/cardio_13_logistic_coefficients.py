# Run the files in numeric order, or use run_all.py.

for name in ["Ridge", "Lasso"]:
    best_pipe = models[name].best_estimator_
    coefs = best_pipe.named_steps["model"].coef_.ravel()

    coef_df = pd.DataFrame({
        "variable": feature_names,
        "coefficient": coefs,
    })
    coef_df["abs_coefficient"] = coef_df["coefficient"].abs()

    print(f"\n{name} — standardized coefficients:")
    display(coef_df.sort_values("abs_coefficient", ascending=False).drop(columns="abs_coefficient"))
