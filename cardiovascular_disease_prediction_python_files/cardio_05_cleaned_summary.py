# Run the files in numeric order, or use run_all.py.

continuous_cols = ["age", "height", "weight", "bmi", "ap_hi", "ap_lo"]
ordinal_binary_cols = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]

print("Summary after cleaning:")
display(df_clean[continuous_cols + ordinal_binary_cols + ["cardio"]].describe().T)

print("\nTarget balance after cleaning:")
display(df_clean["cardio"].value_counts().rename("count").to_frame())
display(df_clean["cardio"].value_counts(normalize=True).rename("proportion").to_frame())
