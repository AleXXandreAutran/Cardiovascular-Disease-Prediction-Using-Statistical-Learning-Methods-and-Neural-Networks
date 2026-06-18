# Run the files in numeric order, or use run_all.py.

feature_cols = [
    "age", "height", "bmi", "ap_hi", "ap_lo",
    "gender", "cholesterol", "gluc", "smoke", "alco", "active",
]
target_col = "cardio"

X = df_clean[feature_cols].copy()
y = df_clean[target_col].astype(int).copy()

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=RANDOM_STATE,
)

numeric_features = ["age", "height", "bmi", "ap_hi", "ap_lo"]
pass_features = ["gender", "cholesterol", "gluc", "smoke", "alco", "active"]
feature_names = numeric_features + pass_features

preprocess = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_features),
        ("pass", "passthrough", pass_features),
    ],
    remainder="drop",
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

print(f"Training set: {X_train.shape}")
print(f"Test set:     {X_test.shape}")
print(f"Positive-class proportion in train: {y_train.mean():.4f}")
print(f"Positive-class proportion in test:  {y_test.mean():.4f}")
