import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_predict
from sklearn.metrics import classification_report, accuracy_score, average_precision_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, make_scorer, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.utils import resample
from imblearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
import shap
import matplotlib.pylab as pl
import time

def load_and_prepare_data(file_name):
    """Load data from a CSV file and prepare it for modeling."""
    data = pd.read_csv(file_name)
    data.drop(columns=['text', 'cluster', 'named_entities'], inplace=True)  # Drop non-numeric or unnecessary columns
    X = data.drop(columns=['performance'])
    y = data['performance']
    return X, y

def train_and_evaluate(X, y):
    """Train the model and evaluate it on the test set, and calculate feature importances."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    feature_importances = model.feature_importances_
    roc_auc_0 = roc_auc_score(y_test == 0, y_proba[:, 0])  # ROC AUC for class 0
    roc_auc_1 = roc_auc_score(y_test == 1, y_proba[:, 1])  # ROC AUC for class
    return accuracy, report, feature_importances, model, roc_auc_0, roc_auc_1

def train_with_smote_kfold(X, y):
    kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    accuracies = []
    global_pr_aucs = []
    global_roc_aucs = []
    global_f1_score = []
    roc_aucs_0 = []
    roc_aucs_1 = []
    pr_aucs_0 = []
    pr_aucs_1 = []
    feature_importances = []
    reports = []

    for train_index, test_index in kf.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        param_test = {
            'max_depth': [3, 5, 7, 10],
            'min_child_weight': [1, 3, 5],
            'gamma': [0, 0.1, 0.2, 0.3],
            'subsample': [0, 0.7, 0.8, 0.9],
            'colsample_bytree': [0, 0.7, 0.8, 0.9]
        }

        # Applying SMOTE
        # smote = SMOTE(random_state=42)
        # X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

        # Adjust the number of neighbors based on the minority class size in the training set
        minority_class_size = min(sum(y_train == 0), sum(y_train == 1))
        n_neighbors = min(minority_class_size - 1, 5)  # at least one or up to 5 neighbors

        if n_neighbors < 1:  # Not enough samples to apply SMOTE
            print("Not enough minority class samples to apply SMOTE. Continuing without SMOTE.")
            X_train_smote, y_train_smote = X_train, y_train
        else:
            smote = SMOTE(random_state=42, k_neighbors=n_neighbors)
            X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)


        # Create a custom scorer. The roc_auc_score will compute the score for class '0' as the positive class
        # minority_auc_scorer = make_scorer(roc_auc_score_minority, response_method="predict_proba")
        #
        # gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=100, objective='binary:logistic',
        #                                                use_label_encoder=False, eval_metric='logloss'),
        #                        param_grid=param_test, scoring=minority_auc_scorer, n_jobs=-1, cv=2)
        #
        # gsearch.fit(X_train_smote, y_train_smote)
        # print(gsearch.best_params_, gsearch.best_score_)

        # Train XGBoost classifier using the best params found in GridSearchCV
        # model = XGBClassifier(use_label_encoder=False, eval_metric='logloss',**gsearch.best_params_)
        model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        model.fit(X_train_smote, y_train_smote)

        # Predictions and Evaluation
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        global_pr_auc = average_precision_score(y_test, y_pred)

        if len(np.unique(y_test)) >= 2:
            global_roc_auc = roc_auc_score(y_test, y_pred)
        else:
            global_roc_auc = None

        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)

        roc_auc_0 = safely_compute_roc_auc(y_test, y_proba, 0)  # ROC AUC for class 0
        roc_auc_1 = safely_compute_roc_auc(y_test, y_proba, 1)  # ROC AUC for class 1
        pr_auc_0 = safely_compute_pr_auc(y_test, y_proba, 0)
        pr_auc_1 = safely_compute_pr_auc(y_test, y_proba, 1)

        accuracies.append(accuracy)
        global_pr_aucs.append(global_pr_auc)
        if global_roc_auc is not None:
            global_roc_aucs.append(global_roc_auc)
        roc_aucs_0.append(roc_auc_0)
        roc_aucs_1.append(roc_auc_1)
        pr_aucs_0.append(pr_auc_0)
        pr_aucs_1.append(pr_auc_1)
        reports.append(report)
        feature_importances.append(model.feature_importances_)

    # Calculate average of the metrics
    avg_accuracy = np.mean(accuracies)
    avg_global_pr_auc = np.mean(global_pr_aucs)
    avg_global_roc_auc = np.mean(global_roc_aucs)
    roc_aucs_0 = [i for i in roc_aucs_0 if i is not None]
    roc_aucs_1 = [i for i in roc_aucs_1 if i is not None]
    pr_aucs_0 = [i for i in pr_aucs_0 if i is not None]
    pr_aucs_1 = [i for i in pr_aucs_1 if i is not None]
    avg_roc_auc_0 = np.mean(roc_aucs_0)
    avg_roc_auc_1 = np.mean(roc_aucs_1)
    avg_pr_auc_0 = np.mean(pr_aucs_0)
    avg_pr_auc_1 = np.mean(pr_aucs_1)
    avg_feature_importances = np.mean(feature_importances, axis=0)

    return (avg_accuracy, avg_roc_auc_0, avg_roc_auc_1, reports, avg_feature_importances,
            avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc, avg_global_roc_auc)

def train_with_smote(X, y):
    # Splitting the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_test = {
        'max_depth': [3, 5, 7, 10],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.2, 0.3],
        'subsample': [0, 0.7, 0.8, 0.9],
        'colsample_bytree': [0, 0.7, 0.8, 0.9]
    }

    # Create a custom scorer. The roc_auc_score will compute the score for class '0' as the positive class
    # minority_auc_scorer = make_scorer(roc_auc_score_minority, response_method="predict_proba")

    #gsearch = GridSearchCV(estimator=XGBClassifier(learning_rate=0.1, n_estimators=100, objective='binary:logistic',
    #                                               use_label_encoder=False, eval_metric='logloss'),
    #                       param_grid=param_test, scoring=minority_auc_scorer, n_jobs=-1, cv=5)


    # Applying SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    #gsearch.fit(X_train_smote, y_train_smote)
    #print(gsearch.best_params_, gsearch.best_score_)

    # Train XGBoost classifier using the best params found in GridSearchCV
    #model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', **gsearch.best_params_)
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train_smote, y_train_smote)

    # Predictions and Evaluation
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    feature_importances = model.feature_importances_

    roc_auc_0 = roc_auc_score(y_test == 0, y_proba[:, 0])  # ROC AUC for class 0
    roc_auc_1 = roc_auc_score(y_test == 1, y_proba[:, 1])  # ROC AUC for class 1
    return accuracy, report, feature_importances, model, roc_auc_0, roc_auc_1


def summarize_results(i, accuracy, report, roc_auc_0, roc_auc_1):
    """Summarize results for output, capturing metrics for both class 0 and class 1."""
    # Extract metrics for each class from the classification report
    class_0_metrics = report.get('0', {})
    class_1_metrics = report.get('1', {})

    return {
        'cluster': i,
        'overall_accuracy': accuracy,  # Overall accuracy is included once
        'precision_0': class_0_metrics.get('precision', 0),
        'recall_0': class_0_metrics.get('recall', 0),
        'roc_auc_0': roc_auc_0,
        'f1-score_0': class_0_metrics.get('f1-score', 0),
        'support_0': class_0_metrics.get('support', 0),
        'precision_1': class_1_metrics.get('precision', 0),
        'recall_1': class_1_metrics.get('recall', 0),
        'roc_auc_1': roc_auc_1,
        'f1-score_1': class_1_metrics.get('f1-score', 0),
        'support_1': class_1_metrics.get('support', 0)
    }

def summarize_results_kfold(i, avg_accuracy, avg_roc_auc_0, avg_roc_auc_1,
                            reports, avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc, global_roc_auc):
    """Summarize results for output, capturing averaged metrics for both class 0 and class 1."""
    # Initialize dictionaries to sum metrics for averaging
    metrics_0 = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    metrics_1 = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}

    # Sum up metrics from each report
    for report in reports:
        if '0' in report:
            for key in metrics_0:
                metrics_0[key] += report['0'][key]
        if '1' in report:
            for key in metrics_1:
                metrics_1[key] += report['1'][key]

    # Average the metrics
    num_reports = len(reports)
    avg_metrics_0 = {key: value / num_reports for key, value in metrics_0.items()}
    avg_metrics_1 = {key: value / num_reports for key, value in metrics_1.items()}

    return {
        'cluster': i,
        'overall_accuracy': avg_accuracy,
        'average_global_pr_auc': avg_global_pr_auc,
        'average_global_roc_auc': global_roc_auc,
        'precision_0': avg_metrics_0['precision'],
        'recall_0': avg_metrics_0['recall'],
        'roc_auc_0': avg_roc_auc_0,
        'pr_auc_0': avg_pr_auc_0,
        'f1-score_0': avg_metrics_0['f1-score'],
        'support_0': avg_metrics_0['support'],
        'precision_1': avg_metrics_1['precision'],
        'recall_1': avg_metrics_1['recall'],
        'roc_auc_1': avg_roc_auc_1,
        'pr_auc_1': avg_pr_auc_1,
        'f1-score_1': avg_metrics_1['f1-score'],
        'support_1': avg_metrics_1['support']
    }

def collect_feature_importances(X_columns, feature_importances, cluster_id):
    """Collect feature importances into a single row for each cluster."""
    # Create a dictionary initializing with the cluster id
    feature_importance_dict = {'cluster': cluster_id}
    # Update the dictionary with feature importances
    feature_importance_dict.update({
        f'{feature}': importance for feature, importance in zip(X_columns, feature_importances)
    })
    return feature_importance_dict


def roc_auc_score_minority(y_true, y_pred):
    return roc_auc_score(y_true, y_pred, average=None, labels=[0])

def safely_compute_roc_auc(y_true, y_proba, pos_label):
    # Check if both classes are present
    if len(np.unique(y_true)) < 2:
        return None  # or another default value, e.g., 0 or the mean ROC AUC of other folds
    return roc_auc_score(y_true == pos_label, y_proba[:, pos_label])


def safely_compute_pr_auc(y_true, y_proba, pos_label):
    if len(np.unique(y_true)) < 2 or np.sum(y_true == pos_label) == 0:
        return None  # or another default value if you prefer
    return average_precision_score(y_true == pos_label, y_proba[:, pos_label])

def main_kfold():
    summary_results = []
    all_feature_importances = []
    for i in range(20):  # Loop from 0_data.csv to 19_data.csv
        start = time.time()
        print(f'Cluster {i}')
        file_name = f'clusters csv\\{i}_data.csv'
        X, y = load_and_prepare_data(file_name)
        (avg_accuracy, avg_roc_auc_0, avg_roc_auc_1, reports, feature_importances ,
         avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc, avg_global_roc_auc) = train_with_smote_kfold(X, y)
        result = summarize_results_kfold(i, avg_accuracy, avg_roc_auc_0, avg_roc_auc_1,
                                         reports, avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc, avg_global_roc_auc)
        summary_results.append(result)
        # Collect feature importances into a single dictionary for each cluster
        feature_importance_dict = collect_feature_importances(X.columns, feature_importances, i)
        all_feature_importances.append(feature_importance_dict)

        end = time.time()
        elapsed = end - start
        print(f'time elapsed for cluster {i}: {elapsed}')

        # SHAP portion, works visually at analysis_notebook.ipynb
        # explainer = shap.TreeExplainer(model)
        # shap_values = explainer.shap_values(X)
        # shap.force_plot(explainer.expected_value, shap_values[0, :], X.columns)

    # Save all results to a single summary CSV file
    results_df = pd.DataFrame(summary_results)
    results_df.to_csv('XGBoost_summary_results.csv', index=False)

    # Combine all feature importance dictionaries into a single DataFrame and save to CSV
    feature_importances_df = pd.DataFrame(all_feature_importances)
    feature_importances_df.to_csv('XGBoost_feature_importances.csv', index=False)

if __name__ == '__main__':
    main_kfold()

    # summary_results = []
    # all_feature_importances = []
    # for i in range(20):  # Loop from 0_data.csv to 19_data.csv
    #     file_name = f'clusters csv\\{i}_data.csv'
    #     X, y = load_and_prepare_data(file_name)
    #     # accuracy, report, feature_importances, model, roc_auc_0, roc_auc_1 = train_and_evaluate(X, y)
    #     accuracy, report, feature_importances, model, roc_auc_0, roc_auc_1 = train_with_smote(X, y)
    #     result = summarize_results(i, accuracy, report, roc_auc_0, roc_auc_1)
    #     summary_results.append(result)
    #     # Collect feature importances into a single dictionary for each cluster
    #     feature_importance_dict = collect_feature_importances(X.columns, feature_importances, i)
    #     all_feature_importances.append(feature_importance_dict)
    #
    #     # SHAP portion, works visually at analysis_notebook.ipynb
    #     # explainer = shap.TreeExplainer(model)
    #     # shap_values = explainer.shap_values(X)
    #     # shap.force_plot(explainer.expected_value, shap_values[0, :], X.columns)
    #
    #
    # # Save all results to a single summary CSV file
    # results_df = pd.DataFrame(summary_results)
    # results_df.to_csv('XGBoost_summary_results.csv', index=False)
    #
    # # Combine all feature importance dictionaries into a single DataFrame and save to CSV
    # feature_importances_df = pd.DataFrame(all_feature_importances)
    # feature_importances_df.to_csv('XGBoost_feature_importances.csv', index=False)
