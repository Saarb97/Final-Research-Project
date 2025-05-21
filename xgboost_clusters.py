import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, accuracy_score, average_precision_score, roc_auc_score
import pandas as pd
import optuna
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from imblearn.under_sampling import TomekLinks
import time
import os
import xgboost as xgb
import torch

# --- Constants ---
# Suppress Optuna info messages, show only warnings/errors
optuna.logging.set_verbosity(optuna.logging.WARNING)
# High number for potential boosting rounds, early stopping will optimize this
MAX_BOOST_ROUNDS = 2000
# Patience for early stopping
EARLY_STOPPING_PATIENCE = 50
# Number of Optuna trials to run
N_OPTUNA_TRIALS = 50 # You might want to reduce this for faster testing initially
VALIDATION_SPLIT_SIZE = 0.2
# Random state for reproducibility
RANDOM_STATE = 42
# Evaluation metric for XGBoost internal early stopping
XGB_EVAL_METRIC = 'auc'
# Optuna optimization direction ('maximize' for AUC/Accuracy, 'minimize' for LogLoss/Error)
OPTUNA_DIRECTION = 'maximize'
# Number of parallel jobs for Optuna
OPTUNA_N_JOBS = 1 # Changed to 1 for easier debugging and to avoid potential resource contention on some systems.
                  # Can be increased if your system handles it well.
N_CV_SPLITS = 5 # Number of splits for StratifiedKFold

# Threshold for minimum samples in a minority class for a cluster to be processed
# Derived from: (N_min * (N_CV_SPLITS-1)/N_CV_SPLITS) * VALIDATION_SPLIT_SIZE >= 1
# and N_min >= N_CV_SPLITS
# For N_CV_SPLITS=5, VALIDATION_SPLIT_SIZE=0.2: (N_min * 4/5) * 0.2 >= 1 => N_min * 0.16 >= 1 => N_min >= 6.25. So, 7.
MIN_SAMPLES_PER_CLASS_THRESHOLD = max(N_CV_SPLITS, int(np.ceil(1 / (VALIDATION_SPLIT_SIZE * (N_CV_SPLITS -1) / N_CV_SPLITS ))) if N_CV_SPLITS > 1 else 1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"CUDA Device Name: {torch.cuda.get_device_name(0)}")


def apply_smote_tomek(X_train, y_train, random_state=42):
    """Applies SMOTE-Tomek to the training data."""
    if not isinstance(y_train, pd.Series): # Ensure y_train is a pd.Series for value_counts()
        y_train_series = pd.Series(y_train)
    else:
        y_train_series = y_train

    # Ensure y_train is integer type for np.bincount and SMOTE
    if not np.issubdtype(y_train_series.dtype, np.integer):
        y_train_series = y_train_series.astype(int)

    class_counts = y_train_series.value_counts()
    if len(class_counts) < 2:
        print("Warning: SMOTE-Tomek received data with only one class. Returning original data.")
        return X_train, y_train_series # Return series

    minority_class_size = class_counts.min()
    # SMOTE's k_neighbors must be less than the number of samples in the minority class.
    # Default k_neighbors for SMOTE is 5.
    n_neighbors_smote = min(minority_class_size - 1, 5) # Using 5 as a common default for k_neighbors in SMOTE

    if n_neighbors_smote < 1:
        print(f"Warning: Not enough minority samples ({minority_class_size}) to apply SMOTE with k_neighbors > 0. Returning original data.")
        return X_train, y_train_series # Return series
    else:
        try:
            # print(f"Applying SMOTE with k_neighbors={n_neighbors_smote}")
            smote = SMOTE(random_state=random_state, k_neighbors=n_neighbors_smote)
            # TomekLinks typically uses sampling_strategy='auto' which targets the majority class by default after oversampling.
            # Or specify 'majority' if you are sure.
            tomek = TomekLinks(sampling_strategy='majority') # Or 'auto'
            smt = SMOTETomek(smote=smote, tomek=tomek, random_state=random_state)

            X_resampled, y_resampled_array = smt.fit_resample(X_train, y_train_series)
            y_resampled = pd.Series(y_resampled_array, name=y_train_series.name) # Keep as series

            # Ensure y_resampled is integer type
            if not np.issubdtype(y_resampled.dtype, np.integer):
                y_resampled = y_resampled.astype(int)
            # print(f"Original class distribution: {class_counts.to_dict()}")
            # print(f"Resampled class distribution: {y_resampled.value_counts().to_dict()}")
            return X_resampled, y_resampled
        except Exception as e:
            print(f"Error during SMOTE-Tomek: {e}. Returning original data.")
            return X_train, y_train_series # Return series

# --- Optuna Objective Function ---
def objective(trial, X_train_fold, y_train_fold):
    """Optuna objective function for XGBoost hyperparameter tuning."""

    # --- 1. Define Hyperparameter Search Space ---
    params = {
        'objective': 'binary:logistic',
        'eval_metric': XGB_EVAL_METRIC,
        'tree_method': 'hist',
        'lambda': trial.suggest_float('lambda', 1e-8, 1.0, log=True),
        'alpha': trial.suggest_float('alpha', 1e-8, 1.0, log=True),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
        'seed': RANDOM_STATE,
        'device': device.type # Pass 'cuda' or 'cpu'
    }

    # --- 2. Split data for this trial's validation and early stopping ---
    # Ensure y_train_fold has enough samples of each class for stratification
    y_train_fold_counts = y_train_fold.value_counts()
    if len(y_train_fold_counts) < 2 or y_train_fold_counts.min() < max(2, int(1/VALIDATION_SPLIT_SIZE) if VALIDATION_SPLIT_SIZE > 0 else 2) : # each class needs enough samples for split
        # This check might be aggressive; Stratified split handles small numbers by warning
        # but we want to avoid single-class validation sets for AUC.
        # We need at least 1 sample of each class in the validation set.
        # If minority in y_train_fold is M, then M * VALIDATION_SPLIT_SIZE >= 1. So M >= 1/VALIDATION_SPLIT_SIZE.
        # For 0.2 split, M >= 5.
        required_min_samples_for_split = int(np.ceil(1/VALIDATION_SPLIT_SIZE)) if VALIDATION_SPLIT_SIZE > 0 else 2
        if y_train_fold_counts.min() < required_min_samples_for_split :
            # print(f"Trial {trial.number}: y_train_fold for Optuna has insufficient minority samples ({y_train_fold_counts.min()}) for stratified val split. Pruning.")
            raise optuna.exceptions.TrialPruned(f"y_train_fold minority too small ({y_train_fold_counts.min()}) for val split.")


    X_sub_train, X_valid, y_sub_train, y_valid = train_test_split(
        X_train_fold, y_train_fold,
        test_size=VALIDATION_SPLIT_SIZE,
        random_state=RANDOM_STATE + trial.number, # Vary random state per trial for robustness
        stratify=y_train_fold
    )

    # ** Crucial check for AUC calculation **
    if len(np.unique(y_valid)) < 2:
        # print(f"Trial {trial.number}: Validation set (y_valid) for early stopping has only one class. Pruning trial.")
        raise optuna.exceptions.TrialPruned("Validation set (y_valid) has only one class.")

    # --- 3. Apply SMOTE-Tomek to the sub-training set ONLY ---
    X_sub_train_resampled, y_sub_train_resampled = apply_smote_tomek(
        X_sub_train, y_sub_train, random_state=RANDOM_STATE + trial.number
    )
    if len(y_sub_train_resampled.value_counts()) < 2: # SMOTE might fail if input y_sub_train was already single class (should be caught earlier)
        # print(f"Trial {trial.number}: Resampled sub-train set has only one class after SMOTE. Pruning.")
        raise optuna.exceptions.TrialPruned("Resampled sub-train set has only one class after SMOTE.")


    # --- 4. Train XGBoost with Early Stopping ---
    model = xgb.XGBClassifier(
        **params,
        n_estimators=MAX_BOOST_ROUNDS,
        early_stopping_rounds=EARLY_STOPPING_PATIENCE,
    )

    try:
        model.fit(
            X_sub_train_resampled, y_sub_train_resampled,
            eval_set=[(X_valid, y_valid)],
            verbose=False
        )
    except xgb.core.XGBoostError as e: # Catch XGBoost specific errors
        if "label set cannot be empty" in str(e) or "Check failed:auc" in str(e):
            # print(f"Error during XGBoost training in trial {trial.number} (likely due to eval_set issue not caught earlier): {e}. Pruning.")
            raise optuna.exceptions.TrialPruned(f"XGBoost fit error: {e}")
        else:
            # print(f"Unhandled XGBoost error during training in trial {trial.number}: {e}")
            return 0.0 if OPTUNA_DIRECTION == 'maximize' else 1e6 # Or re-raise
    except Exception as e:
        # print(f"Generic error during XGBoost training in trial {trial.number}: {e}")
        return 0.0 if OPTUNA_DIRECTION == 'maximize' else 1e6 # Generic failure

    # --- 5. Evaluate on the original validation set ---
    preds_proba = model.predict_proba(X_valid)[:, 1]
    try:
        score = roc_auc_score(y_valid, preds_proba)
    except ValueError as e: # Handles "Only one class present in y_true. ROC AUC score is not defined in that case."
        # print(f"Trial {trial.number}: ValueError calculating ROC AUC on y_valid (should have been caught by earlier checks): {e}. Returning poor score.")
        return 0.0 if OPTUNA_DIRECTION == 'maximize' else 1e6 # Should be caught by the y_valid unique check

    return score


def load_and_prepare_data(file_name, text_col_name, target_col_name):
    """Load data from a CSV file and prepare it for modeling."""
    data = pd.read_csv(file_name)
    # Identify columns to drop more robustly
    cols_to_drop = [text_col_name, 'cluster', 'named_entities']
    existing_cols_to_drop = [col for col in cols_to_drop if col in data.columns]
    if existing_cols_to_drop:
        data.drop(columns=existing_cols_to_drop, inplace=True)

    X = data.drop(columns=[target_col_name])
    y = data[target_col_name]
    return X, y


def train_xgboost_with_SMOTE(X_train_fold, y_train_fold):
    """
    Optimizes XGBoost hyperparameters using Optuna and trains the final model for a specific fold.
    """
    objective_with_data = lambda trial: objective(trial, X_train_fold, y_train_fold)

    # Check if y_train_fold is suitable for Optuna's internal splitting
    y_train_fold_counts = y_train_fold.value_counts()
    required_min_samples_for_optuna_val_split = int(np.ceil(1/VALIDATION_SPLIT_SIZE)) if VALIDATION_SPLIT_SIZE > 0 else 2
    if len(y_train_fold_counts) < 2 or y_train_fold_counts.min() < required_min_samples_for_optuna_val_split:
        print(f"y_train_fold has insufficient minority samples ({y_train_fold_counts.min()}) for Optuna's internal validation split. Skipping Optuna and returning a default model or None.")
        # Fallback: train a default XGBoost model without Optuna or return None
        # For simplicity, we'll return None, and the calling function should handle this.
        # Alternatively, train with default params:
        # default_params = {'objective': 'binary:logistic', 'eval_metric': XGB_EVAL_METRIC, 'tree_method': 'hist', 'seed': RANDOM_STATE, 'device': device.type}
        # model = xgb.XGBClassifier(**default_params, n_estimators=100) # Example default
        # X_train_resampled, y_train_resampled = apply_smote_tomek(X_train_fold, y_train_fold, random_state=RANDOM_STATE)
        # model.fit(X_train_resampled, y_train_resampled)
        # return model
        return None


    study = optuna.create_study(
        direction=OPTUNA_DIRECTION,
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=5),
        sampler=optuna.samplers.TPESampler(seed=RANDOM_STATE)
        # load_if_exists=True # Optional: for resuming
    )
    study.optimize(
        objective_with_data,
        n_trials=N_OPTUNA_TRIALS,
        n_jobs=OPTUNA_N_JOBS,
        timeout=1800
    )
    best_params_from_optuna = study.best_params

    # Prepare final model parameters for this fold using Optuna's best params
    final_model_params = {
        'objective': 'binary:logistic',
        'eval_metric': XGB_EVAL_METRIC,
        'tree_method': 'hist',
        'seed': RANDOM_STATE,
        'device': device.type,
        **best_params_from_optuna
    }

    # For the final model of the fold, we train on the entire X_train_fold (after resampling)
    # and use an internal split for early stopping if desired, or train for best_iteration.

    # Split X_train_fold for early stopping of the final fold model
    # This split is within the current training fold data
    X_train_final_fold_trainpart, X_valid_final_fold_evalpart, \
    y_train_final_fold_trainpart, y_valid_final_fold_evalpart = train_test_split(
        X_train_fold, y_train_fold,
        test_size=VALIDATION_SPLIT_SIZE, # Use the same split ratio
        random_state=RANDOM_STATE + 1000, # Different random state for this split
        stratify=y_train_fold
    )

    # Resample the training part of this final fold model
    X_train_resampled, y_train_resampled = apply_smote_tomek(
        X_train_final_fold_trainpart, y_train_final_fold_trainpart, random_state=RANDOM_STATE
    )

    final_model = xgb.XGBClassifier(
        **final_model_params,
        n_estimators=MAX_BOOST_ROUNDS, # Will be overridden by early stopping if eval_set is valid
        early_stopping_rounds=EARLY_STOPPING_PATIENCE
    )

    eval_set_for_final_fold_model = None
    if len(np.unique(y_valid_final_fold_evalpart)) < 2:
        # Option: Train without early stopping on validation, or use fewer estimators from Optuna if available
        # For now, we'll proceed, and if XGBoost can't use it, it won't do early stopping based on it.
        # Or, explicitly set n_estimators from study.best_trial.user_attrs.get('best_iteration', MAX_BOOST_ROUNDS) if stored
        # For simplicity here, we still provide the eval_set. XGBoost will warn if it's unusable for AUC.
        eval_set_for_final_fold_model = [(X_valid_final_fold_evalpart, y_valid_final_fold_evalpart)]
    else:
        eval_set_for_final_fold_model = [(X_valid_final_fold_evalpart, y_valid_final_fold_evalpart)]


    final_model.fit(
        X_train_resampled, y_train_resampled,
        eval_set=eval_set_for_final_fold_model,
        verbose=False
    )

    return final_model


def train_with_smote_kfold(X, y, cluster_idx):
    kf = StratifiedKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    accuracies = []
    global_pr_aucs = []
    global_roc_aucs = []
    weighted_f1_scores = []
    roc_aucs_0 = []
    roc_aucs_1 = []
    pr_aucs_0 = []
    pr_aucs_1 = []
    feature_importances_list = [] # Changed name
    reports = []

    fold_num = 0
    for train_index, test_index in kf.split(X, y):
        fold_num += 1
        # print(f"-- Starting Fold {fold_num}/{N_CV_SPLITS} for Cluster {cluster_idx} --")
        X_train_fold, X_test_fold = X.iloc[train_index], X.iloc[test_index]
        y_train_fold, y_test_fold = y.iloc[train_index], y.iloc[test_index]

        # Check if y_test_fold is single class, which would make some metrics undefined
        if len(y_test_fold.value_counts()) < 2:
            print(f"Fold {fold_num} for Cluster {cluster_idx}: Test set has only one class. Skipping evaluation for this fold.")
            # Append NaNs or skip appending for this fold's metrics
            # For simplicity, we'll skip appending metrics for this fold if it makes sense for averaging later
            # Or, ensure `safely_compute_roc_auc` etc. handle this by returning None, and np.nanmean handles Nones.
            # We will rely on safely_compute functions and np.nanmean.
            pass # Metrics will be None

        model = train_xgboost_with_SMOTE(X_train_fold, y_train_fold)

        if model is None: # train_xgboost_with_SMOTE might return None if y_train_fold was unsuitable
            print(f"Fold {fold_num} for Cluster {cluster_idx}: Model training skipped due to unsuitable y_train_fold. Metrics for this fold will be NaN.")
            # Append NaNs for all metrics for this fold
            accuracies.append(np.nan)
            global_pr_aucs.append(np.nan)
            global_roc_aucs.append(np.nan)
            weighted_f1_scores.append(np.nan)
            roc_aucs_0.append(np.nan)
            roc_aucs_1.append(np.nan)
            pr_aucs_0.append(np.nan)
            pr_aucs_1.append(np.nan)
            reports.append({}) # Empty report
            # feature_importances_list: handle carefully, maybe append NaNs array of correct shape or skip
            if hasattr(model, 'feature_importances_'): # Should not happen if model is None
                 feature_importances_list.append(model.feature_importances_)
            else: # Append array of NaNs with same shape as X.columns if possible
                 feature_importances_list.append(np.full(X.shape[1], np.nan))
            continue


        # Predictions and Evaluation
        y_pred = model.predict(X_test_fold)
        y_proba = model.predict_proba(X_test_fold)

        accuracy = accuracy_score(y_test_fold, y_pred)
        # For average_precision_score, y_score is y_proba[:, 1] for positive class
        pr_auc_positive_class = safely_compute_pr_auc_sklearn(y_test_fold, y_proba, 1) # Using sklearn's directly if safe
        global_pr_aucs.append(pr_auc_positive_class) # This used y_pred before, should be y_proba for PR AUC

        global_roc_auc_val = safely_compute_roc_auc_sklearn(y_test_fold, y_proba, 1) # For positive class
        global_roc_aucs.append(global_roc_auc_val)

        report = classification_report(y_test_fold, y_pred, output_dict=True, zero_division=0)
        reports.append(report)

        # Compute weighted F1-score
        # Ensure report keys are strings '0' and '1'
        f1_scores_fold = []
        supports_fold = []
        unique_y_test = np.unique(y_test_fold)

        for cls_label in ['0', '1']: # Iterate over possible class labels
            if cls_label in report and int(cls_label) in unique_y_test:
                f1_scores_fold.append(report[cls_label]['f1-score'])
                supports_fold.append(report[cls_label]['support'])
            elif int(cls_label) in unique_y_test: # Class in y_test but not in report (e.g. not predicted)
                 f1_scores_fold.append(0) # F1 is 0
                 supports_fold.append(y_test_fold.value_counts().get(int(cls_label), 0))


        if sum(supports_fold) > 0:
            weighted_f1 = np.average(f1_scores_fold, weights=supports_fold if len(supports_fold) == len(f1_scores_fold) else None)
        else:
            weighted_f1 = np.nan # if no support (e.g. y_test_fold was empty or problematic)
        weighted_f1_scores.append(weighted_f1)


        roc_aucs_0.append(safely_compute_roc_auc(y_test_fold, y_proba, 0))
        roc_aucs_1.append(safely_compute_roc_auc(y_test_fold, y_proba, 1))
        pr_aucs_0.append(safely_compute_pr_auc(y_test_fold, y_proba, 0))
        pr_aucs_1.append(safely_compute_pr_auc(y_test_fold, y_proba, 1))

        accuracies.append(accuracy)
        if hasattr(model, 'feature_importances_'):
            feature_importances_list.append(model.feature_importances_)
        else: # Should not happen if model is not None and fitted
            feature_importances_list.append(np.full(X.shape[1], np.nan))


    # Calculate average of the metrics, ignoring NaNs
    avg_accuracy = np.nanmean(accuracies) if accuracies else np.nan
    avg_global_pr_auc = np.nanmean(global_pr_aucs) if global_pr_aucs else np.nan
    avg_global_roc_auc = np.nanmean(global_roc_aucs) if global_roc_aucs else np.nan # This was for positive class
    avg_weighted_f1_score = np.nanmean(weighted_f1_scores) if weighted_f1_scores else np.nan
    
    avg_roc_auc_0 = np.nanmean(roc_aucs_0) if roc_aucs_0 else np.nan
    avg_roc_auc_1 = np.nanmean(roc_aucs_1) if roc_aucs_1 else np.nan
    avg_pr_auc_0 = np.nanmean(pr_aucs_0) if pr_aucs_0 else np.nan
    avg_pr_auc_1 = np.nanmean(pr_aucs_1) if pr_aucs_1 else np.nan
    
    if feature_importances_list:
        # Check if all elements are np.nan arrays (happens if all folds failed to produce a model)
        is_all_nan_fi = all(np.all(np.isnan(fi_array)) for fi_array in feature_importances_list)
        if is_all_nan_fi:
            avg_feature_importances = np.full(X.shape[1], np.nan)
        else:
            avg_feature_importances = np.nanmean(np.array(feature_importances_list), axis=0)
    else:
        avg_feature_importances = np.full(X.shape[1], np.nan)


    return (avg_accuracy, avg_roc_auc_0, avg_roc_auc_1, reports, avg_feature_importances,
            avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc, avg_global_roc_auc, avg_weighted_f1_score)


def summarize_results_kfold(i, avg_accuracy, avg_roc_auc_0, avg_roc_auc_1,
                            reports, avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc,
                            avg_global_roc_auc, avg_weighted_f1_score, num_features, error_message=None):
    """Summarize results for output, capturing averaged metrics for both class 0 and class 1."""
    if error_message: # If cluster was skipped or had major issues
        summary = {'cluster': i, 'error_message': error_message}
        # Fill other metrics with NaN
        metrics_to_nan = [
            'overall_accuracy', 'average_global_pr_auc', 'average_global_roc_auc', 'average_weighted_f1_score',
            'precision_0', 'recall_0', 'roc_auc_0', 'pr_auc_0', 'f1-score_0', 'support_0',
            'precision_1', 'recall_1', 'roc_auc_1', 'pr_auc_1', 'f1-score_1', 'support_1'
        ]
        for metric_name in metrics_to_nan:
            summary[metric_name] = np.nan
        # Get support from original data if possible, otherwise NaN
        # This part is tricky as y is not directly passed here.
        # For now, if error, support also NaN or handled by caller.
        return summary

    # Initialize dictionaries to sum metrics for averaging
    # These will store sums of metrics from reports
    sum_metrics_0 = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    sum_metrics_1 = {'precision': 0, 'recall': 0, 'f1-score': 0, 'support': 0}
    valid_reports_count = 0

    for report in reports:
        if not report: # Skip empty reports from failed folds
            continue
        valid_reports_count +=1
        if '0' in report:
            for key in sum_metrics_0:
                sum_metrics_0[key] += report['0'].get(key, 0)
        if '1' in report:
            for key in sum_metrics_1:
                sum_metrics_1[key] += report['1'].get(key, 0)

    # Average the metrics from reports
    avg_metrics_0 = {key: (value / valid_reports_count) if valid_reports_count > 0 else np.nan for key, value in sum_metrics_0.items()}
    avg_metrics_1 = {key: (value / valid_reports_count) if valid_reports_count > 0 else np.nan for key, value in sum_metrics_1.items()}

    return {
        'cluster': i,
        'overall_accuracy': avg_accuracy,
        'average_global_pr_auc': avg_global_pr_auc, # This is for positive class (1) typically
        'average_global_roc_auc': avg_global_roc_auc, # This is for positive class (1) typically
        'average_weighted_f1_score': avg_weighted_f1_score,
        'precision_0': avg_metrics_0['precision'],
        'recall_0': avg_metrics_0['recall'],
        'roc_auc_0': avg_roc_auc_0, # Already averaged k-fold ROC AUC for class 0
        'pr_auc_0': avg_pr_auc_0,   # Already averaged k-fold PR AUC for class 0
        'f1-score_0': avg_metrics_0['f1-score'],
        'support_0': avg_metrics_0['support'], # This is average support per fold
        'precision_1': avg_metrics_1['precision'],
        'recall_1': avg_metrics_1['recall'],
        'roc_auc_1': avg_roc_auc_1, # Already averaged k-fold ROC AUC for class 1
        'pr_auc_1': avg_pr_auc_1,   # Already averaged k-fold PR AUC for class 1
        'f1-score_1': avg_metrics_1['f1-score'],
        'support_1': avg_metrics_1['support'], # This is average support per fold
        'error_message': None
    }


def collect_feature_importances(X_columns, feature_importances_array, cluster_id, error_message=None):
    """Collect feature importances into a single row for each cluster."""
    feature_importance_dict = {'cluster': cluster_id, 'error_message': error_message}
    if error_message is None and feature_importances_array is not None and not np.all(np.isnan(feature_importances_array)):
        feature_importance_dict.update({
            feature: importance for feature, importance in zip(X_columns, feature_importances_array)
        })
    else: # Fill with NaNs if error or no importances
        for feature in X_columns:
            feature_importance_dict[feature] = np.nan
    return feature_importance_dict


def safely_compute_roc_auc(y_true, y_proba, class_label):
    """Computes ROC AUC for a specific class. y_proba is the probability array for all classes."""
    if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true)
    y_true_binary = (y_true == class_label).astype(int)
    if len(np.unique(y_true_binary)) < 2: # Check if the derived binary target has both classes
        # print(f"Warning: ROC AUC for class {class_label} cannot be computed. y_true_binary has only one class: {np.unique(y_true_binary)}")
        return np.nan
    if y_proba.shape[1] <= class_label:
        # print(f"Warning: class_label {class_label} is out of bounds for y_proba columns {y_proba.shape[1]}")
        return np.nan
    return roc_auc_score(y_true_binary, y_proba[:, class_label])

def safely_compute_pr_auc(y_true, y_proba, class_label):
    """Computes PR AUC for a specific class."""
    if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true)
    y_true_binary = (y_true == class_label).astype(int)
    if np.sum(y_true_binary) == 0: # No positive samples for this class
        # print(f"Warning: PR AUC for class {class_label} cannot be computed. No true samples for this class.")
        return np.nan
    if len(np.unique(y_true_binary)) < 2 and np.sum(y_true_binary) == len(y_true_binary) : # All samples are of this class
        # print(f"Warning: PR AUC for class {class_label} cannot be computed. All true samples are of this class.")
        return np.nan # Or 1.0, depending on definition, but average_precision_score might error.
    if y_proba.shape[1] <= class_label:
        # print(f"Warning: class_label {class_label} is out of bounds for y_proba columns {y_proba.shape[1]}")
        return np.nan
    return average_precision_score(y_true_binary, y_proba[:, class_label])

# Wrapper for sklearn's roc_auc_score if y_true is already multiclass and y_score is for positive class
def safely_compute_roc_auc_sklearn(y_true, y_proba_all_classes, positive_class_label=1):
    if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true)
    if len(y_true.value_counts()) < 2 :
        return np.nan
    return roc_auc_score(y_true, y_proba_all_classes[:, positive_class_label]) # Assumes positive_class_label is 1 for binary

def safely_compute_pr_auc_sklearn(y_true, y_proba_all_classes, positive_class_label=1):
    if not isinstance(y_true, pd.Series): y_true = pd.Series(y_true)
    if len(y_true.value_counts()) < 2 or y_true.value_counts().get(positive_class_label, 0) == 0:
        return np.nan
    return average_precision_score(y_true, y_proba_all_classes[:, positive_class_label])


def main_kfold(data_files_loc, num_of_clusters, output_loc, text_col_name, target_col_name):
    os.makedirs(output_loc, exist_ok=True) # Ensure output directory exists
    summary_results_list = [] # Renamed from summary_results
    all_feature_importances_list = [] # Renamed

    # Get a template for X.columns for skipped clusters, if any data file exists
    X_columns_template = None
    try:
        # Try to load columns from the first expected file to use as template for skipped ones
        temp_file_name = os.path.join(data_files_loc, f'{0}_data.csv')
        if os.path.exists(temp_file_name):
            X_temp, _ = load_and_prepare_data(temp_file_name, text_col_name, target_col_name)
            X_columns_template = X_temp.columns
    except Exception as e:
        print(f"Could not load template columns from 0_data.csv: {e}")


    for i in range(num_of_clusters):
        start_time = time.time() # Renamed from start
        print(f"\nProcessing Cluster {i}...")
        file_name = os.path.join(data_files_loc, f'{i}_data.csv')

        if not os.path.exists(file_name):
            print(f"File {file_name} not found for cluster {i}. Skipping.")
            error_msg = "Data file not found"
            # Use X_columns_template if available, otherwise an empty list/None
            # which `collect_feature_importances` and `summarize_results_kfold` should handle
            summary_results_list.append(
                summarize_results_kfold(i, np.nan, np.nan, np.nan, [], np.nan, np.nan, np.nan, np.nan, np.nan,
                                        len(X_columns_template) if X_columns_template is not None else 0,
                                        error_message=error_msg)
            )
            all_feature_importances_list.append(
                collect_feature_importances(X_columns_template if X_columns_template is not None else [], None, i, error_message=error_msg)
            )
            continue

        X, y = load_and_prepare_data(file_name, text_col_name, target_col_name)
        if X_columns_template is None: # Set template if first successful load
             X_columns_template = X.columns

        class_counts = y.value_counts()
        print(f"Cluster {i} raw class counts: {class_counts.to_dict()}")

        error_msg = None
        if len(class_counts) <= 1:
            error_msg = "Target variable has only one class"
        elif class_counts.min() < MIN_SAMPLES_PER_CLASS_THRESHOLD:
            error_msg = f"Minority class too small ({class_counts.min()}), threshold is {MIN_SAMPLES_PER_CLASS_THRESHOLD}"

        if error_msg:
            print(f"Skipping cluster {i}: {error_msg}.")
            summary_results_list.append(
                summarize_results_kfold(i, np.nan, np.nan, np.nan, [], np.nan, np.nan, np.nan, np.nan, np.nan,
                                        X.shape[1], error_message=error_msg)
            )
            all_feature_importances_list.append(
                collect_feature_importances(X.columns, None, i, error_message=error_msg)
            )
            elapsed_time = time.time() - start_time
            print(f"Time elapsed for cluster {i} (skipped): {elapsed_time:.2f} seconds")
            continue

        #  print(f"Training model for Cluster {i} with {N_CV_SPLITS}-fold CV...")
        (avg_accuracy, avg_roc_auc_0, avg_roc_auc_1, reports, current_feature_importances, # Renamed
         avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc,
         avg_global_roc_auc, avg_weighted_f1_score) = train_with_smote_kfold(X, y, i)

        result_summary = summarize_results_kfold( # Renamed
            i, avg_accuracy, avg_roc_auc_0, avg_roc_auc_1, reports,
            avg_pr_auc_0, avg_pr_auc_1, avg_global_pr_auc, avg_global_roc_auc, avg_weighted_f1_score, X.shape[1]
        )
        summary_results_list.append(result_summary)

        feature_importance_dict = collect_feature_importances(X.columns, current_feature_importances, i) # Renamed
        all_feature_importances_list.append(feature_importance_dict)

        elapsed_time = time.time() - start_time # Renamed
        print(f"Time elapsed for cluster {i}: {elapsed_time:.2f} seconds")
        #  print(f"Cluster {i} Results: Accuracy={avg_accuracy:.4f}, WeightedF1={avg_weighted_f1_score:.4f} GlobalROCAUC={avg_global_roc_auc:.4f}")


    # Save all results to a single summary CSV file
    results_df = pd.DataFrame(summary_results_list)
    results_df.to_csv(os.path.join(output_loc, 'XGBoost_summary_results.csv'), index=False)
    print(f"\nSaved summary results to {os.path.join(output_loc, 'XGBoost_summary_results.csv')}")

    # Combine all feature importance dictionaries into a single DataFrame and save to CSV
    if all_feature_importances_list: # Ensure list is not empty
        feature_importances_df = pd.DataFrame(all_feature_importances_list)
        feature_importances_df.to_csv(os.path.join(output_loc, 'XGBoost_feature_importances.csv'), index=False)
        print(f"Saved feature importances to {os.path.join(output_loc, 'XGBoost_feature_importances.csv')}")
    else:
        print("No feature importances were generated (e.g., all clusters skipped).")


# if __name__ == '__main__':
#     # --- Configuration for main_kfold ---
#     # These would typically come from command-line arguments or a config file
#     DATA_FILES_LOCATION = 'clusters_csv' # Example: 'path/to/your/cluster_csv_files'
#     NUMBER_OF_CLUSTERS = 20              # Example: 20 clusters (0_data.csv to 19_data.csv)
#     OUTPUT_LOCATION = 'XGBoost_Output'   # Example: 'path/to/output_directory'
#     TEXT_COLUMN_NAME = 'text'            # The name of the text column to be dropped
#     TARGET_COLUMN_NAME = 'target'        # The name of your target variable column

#     # Create dummy data for testing if it doesn't exist
#     if not os.path.exists(DATA_FILES_LOCATION):
#         print(f"Creating dummy data in '{DATA_FILES_LOCATION}' for testing...")
#         os.makedirs(DATA_FILES_LOCATION, exist_ok=True)
#         num_features = 10
#         feature_names = [f'feature_{k}' for k in range(num_features)]
#         for i in range(NUMBER_OF_CLUSTERS):
#             file_path = os.path.join(DATA_FILES_LOCATION, f'{i}_data.csv')
#             num_samples = np.random.randint(5, 150) # Vary sample sizes
            
#             X_data = np.random.rand(num_samples, num_features)
#             df = pd.DataFrame(X_data, columns=feature_names)
#             df[TEXT_COLUMN_NAME] = [f"dummy text {j}" for j in range(num_samples)]
#             df['cluster'] = i # Add cluster column for consistency with drop logic
            
#             # Create different imbalance scenarios
#             if i % 4 == 0: # Highly imbalanced - triggers skip
#                 y_data = np.array([0]* (num_samples - 2) + [1]*2)
#             elif i % 4 == 1: # Moderately imbalanced - might be borderline
#                  minority_count = max(MIN_SAMPLES_PER_CLASS_THRESHOLD -1 , int(num_samples * 0.1))
#                  y_data = np.array([0]* (num_samples - minority_count) + [1]*minority_count)
#             elif i % 4 == 2: # Balanced
#                 y_data = np.array([0]* (num_samples // 2) + [1]*(num_samples - num_samples // 2) )
#             else: # Single class - triggers skip
#                 y_data = np.zeros(num_samples, dtype=int)

#             np.random.shuffle(y_data)
#             df[TARGET_COLUMN_NAME] = y_data
#             df.to_csv(file_path, index=False)
#         print(f"Dummy data creation complete. Check '{DATA_FILES_LOCATION}'.")
#         print(f"MIN_SAMPLES_PER_CLASS_THRESHOLD is set to: {MIN_SAMPLES_PER_CLASS_THRESHOLD}")


#     main_kfold(
#         data_files_loc=DATA_FILES_LOCATION,
#         num_of_clusters=NUMBER_OF_CLUSTERS,
#         output_loc=OUTPUT_LOCATION,
#         text_col_name=TEXT_COLUMN_NAME,
#         target_col_name=TARGET_COLUMN_NAME
#     )