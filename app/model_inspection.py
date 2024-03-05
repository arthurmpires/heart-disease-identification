import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn as sns
import dataframe_image as dfi

def save_selected_hyperparameters(study):
    selected_hyperparameters = pd.DataFrame.from_dict(study.best_params, orient="index", columns=["Value"])
    dfi.export(selected_hyperparameters, "../reports/selected_hyperparameters.png")

def save_regression_coefficients(model, x_train):
    fig, ax = plt.subplots(figsize=(15,8))
    g = sns.barplot(pd.Series(model[-1].coef_[0], x_train.columns).sort_values(), orient="h", ax=ax)
    g.set_title("Regression Coefficients")
    plt.savefig("../reports/regression_coeffients.png")

def save_class_metrics(y_test, y_pred):
    report_metrics = pd.DataFrame(metrics.classification_report(y_test, y_pred, output_dict=True))[["Absence", "Presence"]]
    report_metrics = report_metrics.map(lambda x: format(x, ".2f") if x < 1 else format(x, ".0f"))
    report_metrics.dfi.export("../reports/class_metrics.png")

def save_roc_curve(y_true, y_prob):
    metrics.RocCurveDisplay.from_predictions(y_true, y_prob)
    plt.title("ROC Curve")
    plt.savefig("../reports/ROC_curve.png")

def save_confusion_matrix(y_test, y_pred):
    metrics.ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.title("Confusion Matrix")
    plt.savefig("../reports/confusion_matrix.png")