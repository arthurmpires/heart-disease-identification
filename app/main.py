from app import read_write, model_training, model_inspection

# download data
read_write.download_raw_data()

# train model
heart_disease = read_write.open_dataset()
x_train, x_test, y_train, y_test = model_training.create_training_sets(heart_disease)
study = model_training.select_model(x_train, y_train)
model, y_prob, y_true, y_pred = model_training.train_model(study, x_train, x_test, y_train, y_test)
read_write.save_training_artifacts(study, model, x_train, x_test, y_train, y_test, y_prob, y_true, y_pred)

# inspect model
model_inspection.save_selected_hyperparameters(study)
model_inspection.save_regression_coefficients(model, x_train)
model_inspection.save_class_metrics(y_test, y_pred)
model_inspection.save_roc_curve(y_true, y_prob)
model_inspection.save_confusion_matrix(y_test, y_pred)