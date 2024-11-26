import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
import time

def load_and_prepare_data(file_name):
    """Load data from a CSV file and prepare it for modeling."""
    data = pd.read_csv(file_name)

    # Define the columns to drop before analysing results
    columns_to_drop = ['text', 'cluster', 'named_entities', 'Unnamed: 0']

    # Filter columns that exist in the data
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=existing_columns_to_drop, inplace=True)

    X = data.drop(columns=['performance'])
    y = data['performance']
    return X, y


if __name__ == '__main__':
    data_file_name = f'full_dataset_feature_extraction_09-05.csv'
    X, y = load_and_prepare_data(data_file_name)

    min_features_to_select = 1  # Minimum number of features to consider
    clf = LogisticRegression(max_iter=10000, solver='saga')
    cv = StratifiedKFold(5)

    rfecv = RFECV(
        estimator=clf,
        step=1,
        cv=cv,
        scoring="accuracy",
        min_features_to_select=min_features_to_select,
        n_jobs=-1,
    )
    start = time.time()
    rfecv.fit(X, y)
    end = time.time()
    elapsed = end - start
    print(f'elapsed time: {elapsed}')

    print(f"Optimal number of features: {rfecv.n_features_}")
    print(f"features: {rfecv.get_feature_names_out()}")

