# Run the files in numeric order, or use run_all.py.

df = df_raw.copy()

print("Variable types:")
display(df.dtypes)

print("\nMissing values:")
display(df.isna().sum())

print("\nInitial target distribution:")
display(df["cardio"].value_counts(normalize=True).rename("proportion").to_frame())

display(df.describe(include="all").T)
