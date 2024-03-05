import pandas as pd
import joblib

def download_raw_data():
    heart_disease = pd.read_parquet("https://openml1.win.tue.nl/datasets/0004/43823/dataset_43823.pq")
    heart_disease.to_parquet("../data/raw/heart_disease.pq")

def open_dataset():
    heart_disease = pd.read_parquet("../data/raw/heart_disease.pq")
    return heart_disease

def save_training_artifacts(study, model, x_train, x_test, y_train, y_test, y_prob, y_true, y_pred):
    joblib.dump(study, "../models/heart_disease_study.pkl")
    joblib.dump(model, "../models/heart_disease_model.pkl")
    x_train.to_parquet("../data/processed/x_train.pq")
    x_test.to_parquet("../data/processed/x_test.pq")
    y_train.to_csv("../data/processed/y_train.csv", index=False)
    y_test.to_csv("../data/processed/y_test.csv", index=False)
    y_prob.to_csv("../models/y_prob.csv", index=False)
    y_true.to_csv("../models/y_true.csv", index=False)
    y_pred.to_csv("../models/y_pred.csv", index=False)

def read_inspection_artifacts():
    study = joblib.load("../models/heart_disease_study.pkl")
    model = joblib.load("../models/heart_disease_model.pkl")
    x_train = pd.read_parquet("../data/processed/x_train.pq")
    x_test = pd.read_parquet("../data/processed/x_test.pq")
    y_train = pd.read_csv("../data/processed/y_train.csv")
    y_test = pd.read_csv("../data/processed/y_test.csv")
    y_prob = pd.read_csv("../models/y_prob.csv")
    y_true = pd.read_csv("../models/y_true.csv")
    y_pred = pd.read_csv("../models/y_pred.csv")
    return study, model, x_train, x_test, y_train, y_test, y_prob, y_true, y_pred


