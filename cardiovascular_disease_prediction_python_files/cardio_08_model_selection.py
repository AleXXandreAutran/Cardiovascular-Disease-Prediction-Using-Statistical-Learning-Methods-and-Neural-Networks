# Run the files in numeric order, or use run_all.py.

def make_grid_search(estimator, param_grid, name: str) -> GridSearchCV:
    """Create, fit, and return a cross-validated pipeline."""
    pipe = Pipeline([
        ("preprocess", preprocess),
        ("model", estimator),
    ])

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        refit=True,
        return_train_score=True,
    )

    print(f"Training: {name}")
    grid.fit(X_train, y_train)
    print(f"  best mean CV AUC = {grid.best_score_:.4f}")
    print(f"  best parameters  = {grid.best_params_}\n")
    return grid


models = {}

models["Ridge"] = make_grid_search(
    LogisticRegression(penalty="l2", solver="lbfgs", max_iter=5000, random_state=RANDOM_STATE),
    {"model__C": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
    "Ridge logistic regression",
)

models["Lasso"] = make_grid_search(
    LogisticRegression(penalty="l1", solver="liblinear", max_iter=5000, random_state=RANDOM_STATE),
    {"model__C": [1e-3, 1e-2, 1e-1, 1, 10, 100]},
    "Lasso logistic regression",
)

models["KNN"] = make_grid_search(
    KNeighborsClassifier(algorithm="auto", n_jobs=-1),
    {"model__n_neighbors": [20, 50, 100], "model__weights": ["uniform"]},
    "k-nearest neighbors",
)

models["Random forest"] = make_grid_search(
    RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
    {
        "model__n_estimators": [100, 300],
        "model__max_depth": [8, 12, None],
        "model__min_samples_leaf": [5, 10],
    },
    "Random forest",
)

models["MLP"] = make_grid_search(
    MLPClassifier(
        activation="relu",
        solver="adam",
        early_stopping=True,
        max_iter=300,
        random_state=RANDOM_STATE,
    ),
    {
        "model__hidden_layer_sizes": [(50,), (100,), (50, 25)],
        "model__alpha": [1e-4, 1e-3],
        "model__learning_rate_init": [1e-3],
    },
    "Multilayer perceptron",
)
