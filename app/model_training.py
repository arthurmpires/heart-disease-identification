from sklearn import model_selection, pipeline, preprocessing, linear_model, metrics
import optuna
import pandas as pd

def create_training_sets(dataset):
    x = dataset.drop(columns=["Heart_Disease"])
    y = dataset["Heart_Disease"]
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=.2, random_state=42)
    return x_train, x_test, y_train, y_test

def objective(trial, x_train, y_train):
    class_weight = trial.suggest_categorical("class_weight", ["balanced", None])
    penalty = trial.suggest_categorical("penalty", ["l1", "l2", "elasticnet", None])
    if penalty is None:
        clf = linear_model.LogisticRegression(solver='lbfgs', class_weight=class_weight, penalty=penalty)
    else:
        C = trial.suggest_float("C", 1e-2, 1e3, log=True)
        if penalty == "elasticnet":
            l1_ratio = trial.suggest_float("l1_ratio", 0, 1)
            clf = linear_model.LogisticRegression(solver="saga", class_weight=class_weight, penalty=penalty, C=C, l1_ratio=l1_ratio)
        else:
            clf = linear_model.LogisticRegression(solver="liblinear", class_weight=class_weight, penalty=penalty, C=C)
    model = pipeline.Pipeline([("scaler", preprocessing.StandardScaler()), ("classifier", clf)])
    score = model_selection.cross_val_score(model, x_train, y_train, scoring="roc_auc", cv=4).mean()
    return score

def select_model(x_train, y_train):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(study_name="Logistic Regression for Heart Disease Prediction", direction="maximize")
    study.optimize(lambda trial: objective(trial, x_train, y_train), n_trials=40, show_progress_bar=True)
    return study

def train_model(study, x_train, x_test, y_train, y_test):
    clf = \
        linear_model.LogisticRegression(solver="lbfgs", **study.best_params) if study.best_params["penalty"] is None else \
        linear_model.LogisticRegression(solver="saga", **study.best_params) if study.best_params["penalty"] == "elasticnet" else \
        linear_model.LogisticRegression(solver="liblinear", **study.best_params)
    model = pipeline.Pipeline([("scaler", preprocessing.StandardScaler()), ("classifier", clf)]).fit(x_train, y_train)
    y_prob = pd.Series(model.predict_proba(x_test)[:, 1])
    y_true = y_test.map({"Absence": 0, "Presence": 1})
    _, tpr, thresholds = metrics.roc_curve(y_true, y_prob)
    threshold = pd.Series(thresholds)[pd.Series(tpr).loc[tpr > .90].index.min()+1]
    y_pred = pd.Series(["Presence" if prob > threshold else "Absence" for prob in y_prob])
    return model, y_prob, y_true, y_pred