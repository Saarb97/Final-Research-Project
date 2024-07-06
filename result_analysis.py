import pandas as pd
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.linear_model import LogisticRegression
import numpy as np
from xgboost_clusters import train_xgboost_with_SMOTE
from sklearn.model_selection import train_test_split
import shap

def load_and_prepare_data(file_name):
    """Load data from a CSV file and prepare it for modeling."""
    data = pd.read_csv(file_name)
    try:
        data.drop(columns=['text', 'cluster', 'named_entities', 'Unnamed: 0'], inplace=True)  # Drop non-numeric or unnecessary columns
    except:
        data.drop(columns=['text', 'cluster', 'Unnamed: 0'], inplace=True)  # Drop non-numeric or unnecessary columns
    return data

def train_and_evaluate(X, y):
    """Train the model and evaluate it on the test set, and calculate feature importances."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = train_xgboost_with_SMOTE(X_train, y_train)
    return model

def get_shap_feature_importance(data_file_name):
    data = load_and_prepare_data(data_file_name)
    X = data.drop(columns=['performance'])
    y = data['performance']
    model = train_and_evaluate(X, y)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)

    # Get model predictions
    y_pred = model.predict(X)

    # Filter indices where the model predicts class 0 and the true label is also 0 (True Negatives)
    true_negatives_indices = (y_pred == 0) & (y == 0)

    # Get the indices of true negatives
    class_0_indices = true_negatives_indices[true_negatives_indices].index

    # Get the corresponding SHAP values for the samples predicted as class 0
    class_0_shap_values = shap_values[class_0_indices]

    # Calculate mean absolute SHAP values for each feature
    class_0_mean_shap_values = np.mean(class_0_shap_values, axis=0)
    shap_importance_df = pd.DataFrame({
        'Feature': X.columns,
        'SHAP_Importance': class_0_mean_shap_values
    }).sort_values(ascending=True, by=['SHAP_Importance'])

    return shap_importance_df

def calc_statistical_information(data_df, test_type):
    # Initialize dictionaries to store the mean values and test results
    mean_0 = {}
    mean_1 = {}
    t_statistics = {}
    p_values = {}
    significant = {}

    # Loop through each column except 'performance'
    for column in data_df.columns:
        if column != 'performance':
            # Calculate means
            mean_0[column] = data_df[data_df['performance'] == 0][column].mean()
            mean_1[column] = data_df[data_df['performance'] == 1][column].mean()

            # Perform test
            group_0 = data_df[data_df['performance'] == 0][column]
            group_1 = data_df[data_df['performance'] == 1][column]

            if group_0.var() > 0 and group_1.var() > 0:
                if test_type == 't-test':
                    t_stat, p_val = ttest_ind(group_0, group_1, nan_policy='omit')
                elif test_type == 'Mann-Whitney U test':
                    t_stat, p_val = mannwhitneyu(group_0, group_1, alternative='two-sided')
                else:
                    print('Unknown test, expected "t-test" or "Mann-Whitney U test", using Mann-Whitney as default')
                    t_stat, p_val = mannwhitneyu(group_0, group_1, alternative='two-sided')
            else:
                t_stat, p_val = float('nan'), float('nan')

            # Store test results
            t_statistics[column] = t_stat
            p_values[column] = p_val
            significant[column] = p_val < 0.05

    # Create a new DataFrame to store the mean values
    mean_df = pd.DataFrame([mean_0, mean_1], index=['mean_0', 'mean_1'])

    # Append test results to the DataFrame
    mean_df.loc['t_statistic'] = t_statistics
    mean_df.loc['p_value'] = p_values
    mean_df.loc['significant'] = significant

    return mean_df

def run_logistic_regression(X, y, num_of_features):
    log_reg = LogisticRegression(max_iter=10000, solver='saga').fit(X, y)

    coefficients = log_reg.coef_[0]
    intercept = log_reg.intercept_[0]

    # Calculate the odds ratio for each feature
    odds_ratios = np.exp(coefficients)

    # Convert odds ratios to probability increase
    prob_increase = (odds_ratios - 1) * 100

    # Create a DataFrame to display the results
    feature_effects = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': coefficients,
        'Odds Ratio': odds_ratios,
        'Probability Increase (%)': prob_increase
    })

    # Sort the DataFrame by the probability increase (lowest to highest)
    sorted_features = feature_effects.sort_values(by='Probability Increase (%)').head(num_of_features)
    sorted_features['Probability Increase (%)'] = sorted_features['Probability Increase (%)'].abs()

    for index, row in sorted_features.iterrows():
        if row['Probability Increase (%)'] > 2:
            print(
                f"A sample with high indication of {row['Feature']} has {abs(row['Probability Increase (%)']):.2f}% higher chance to be in class 0.")

    return sorted_features

if __name__ == '__main__':

    for i in range(20):  # Loop from 0_data.csv to 19_data.csv
        print(f'Cluster {i}')
        data_file_name = f'clusters csv\\{i}_data.csv'
        statistics_file_name = f'clusters csv\\{i}_statistics.csv'
        data_df = load_and_prepare_data(data_file_name)
        statistics_df = pd.read_csv(statistics_file_name)

        test_types = ['t-test', 'Mann-Whitney U test']
        mean_df = calc_statistical_information(data_df, test_types[1])

        # Display the combined DataFrame
        #print(mean_df)
        #mean_df.to_csv(f'clusters csv\\{i}_results_analysis.csv', index=True)

        # Filter the significant features
        significant_features = mean_df.columns[mean_df.loc['significant'] == 1]
        # print(significant_features)
        # Separate features and target
        X = data_df[significant_features]
        y = data_df['performance']

        num_of_features = 10
        statistical_important_features = run_logistic_regression(X, y, num_of_features)
        statistical_important_features.to_csv(f'results\\{i}_statistical.csv', index=False)

        shap_feature_importance = get_shap_feature_importance(data_file_name).head(num_of_features)
        #print(shap_feature_importance)
        shap_feature_importance.to_csv(f'results\\{i}_shap.csv', index=False)




