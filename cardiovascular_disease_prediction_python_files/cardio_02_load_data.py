# Run the files in numeric order, or use run_all.py.

def load_dataset(path: Path) -> pd.DataFrame:
    """Load the cardiovascular disease dataset from a CSV file."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Put the CSV file in the data/ folder "
            "or change DATA_PATH at the top of the notebook."
        )

    # sep=None lets pandas infer whether the file is comma- or semicolon-separated.
    df = pd.read_csv(path, sep=None, engine="python", decimal=",")
    print(f"File loaded: {path}")
    print(f"Initial shape: {df.shape}")
    return df


df_raw = load_dataset(DATA_PATH)
df_raw.head()
