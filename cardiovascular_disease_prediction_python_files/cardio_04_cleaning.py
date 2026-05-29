# Generated from Code_cardiovascular_disease_prediction.ipynb
# One file corresponds to one non-empty code cell from the notebook.
# Run the files in numeric order, or use run_all.py.

def clean_cardio_data(df_in: pd.DataFrame) -> pd.DataFrame:
    """Clean the cardiovascular dataset using deterministic rules."""
    df = df_in.copy()

    if "id" in df.columns:
        df = df.drop(columns="id")

    numeric_cols = ["age", "height", "weight", "ap_hi", "ap_lo"]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Age is provided in days in the original Kaggle dataset.
    df["age"] = df["age"] / 365.25

    # Correct obvious sign errors in blood pressure.
    df["ap_hi"] = df["ap_hi"].abs()
    df["ap_lo"] = df["ap_lo"].abs()

    # Correct simple scale errors.
    df.loc[df["ap_hi"] >= 1000, "ap_hi"] /= 10
    df.loc[df["ap_lo"] >= 1000, "ap_lo"] /= 10
    df.loc[df["ap_hi"].between(11, 25), "ap_hi"] *= 10
    df.loc[df["ap_lo"].between(4, 15), "ap_lo"] *= 10

    # Swap systolic and diastolic pressure when the swap gives a plausible pair.
    mask_swap = (
        (df["ap_lo"] > df["ap_hi"])
        & df["ap_lo"].between(60, 250)
        & df["ap_hi"].between(40, 150)
    )
    df.loc[mask_swap, ["ap_hi", "ap_lo"]] = df.loc[mask_swap, ["ap_lo", "ap_hi"]].to_numpy()

    # BMI in kg/m^2.
    df["bmi"] = df["weight"] / (df["height"] / 100) ** 2

    # Physiological plausibility filters.
    mask_keep = (
        df["age"].between(18, 100)
        & df["height"].between(130, 210)
        & df["weight"].between(40, 200)
        & df["bmi"].between(15, 50)
        & df["ap_hi"].between(60, 250)
        & df["ap_lo"].between(40, 150)
        & (df["ap_hi"] >= df["ap_lo"])
        & df["cardio"].isin([0, 1])
    )

    df = df.loc[mask_keep].reset_index(drop=True)

    # Ensure integer-like categorical variables remain numeric.
    for col in ["gender", "cholesterol", "gluc", "smoke", "alco", "active", "cardio"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(int)

    return df


df_clean = clean_cardio_data(df_raw)

print(f"Initial number of observations: {len(df_raw):,}")
print(f"Final number of observations:   {len(df_clean):,}")
print(f"Removed observations:          {len(df_raw) - len(df_clean):,}")
df_clean.head()
