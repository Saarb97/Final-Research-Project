import numpy as np
import pandas as pd
from sklearn.ensemble._iforest import _average_path_length
import numpy as np
import multiprocessing
from functools import partial
from sklearn.ensemble import IsolationForest
from causallearn.search.FCMBased import lingam
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu







if __name__ == '__main__':
    FILE_PATH = "full_dataset_feature_extraction_12-03-24.csv"
    data_df = pd.read_csv(FILE_PATH)

    # Group data by cluster
    cluster_groups = data_df.groupby('cluster')

    # Get the dataframe of passed prompts (performance=1, cluster=-1)
    passed_prompts = data_df[(data_df['performance'] == 1) & (data_df['cluster'] == -1)].drop(
        ['cluster', 'performance', 'text'], axis=1)

    # Create a dictionary to store dataframes for each cluster
    cluster_dfs = {}

    # Iterate over each cluster group
    for cluster, group in cluster_groups:
        # Create a new dataframe for the current cluster
        cluster_df = pd.DataFrame()

        # Calculate descriptive statistics
        cluster_df = cluster_df.assign(mean=group.drop(['text', 'cluster', 'performance'], axis=1).mean(),
                                       median=group.drop(['text', 'cluster', 'performance'], axis=1).median(),
                                       std=group.drop(['text', 'cluster', 'performance'], axis=1).std(),
                                       min=group.drop(['text', 'cluster', 'performance'], axis=1).min(),
                                       max=group.drop(['text', 'cluster', 'performance'], axis=1).max())

        # Calculate statistical significance (Mann-Whitney U Test)
        u_stats = []
        p_vals = []
        for col in group.drop(['text', 'cluster', 'performance'], axis=1).columns:
            u_stat, p_val = mannwhitneyu(group[col], passed_prompts[col], alternative='two-sided')
            u_stats.append(u_stat)
            p_vals.append(p_val)

        cluster_df = cluster_df.assign(u_stat=u_stats, p_val=p_vals)

        cluster_dfs[cluster] = cluster_df

    # Print the dataframes for each cluster
    for cluster, df in cluster_dfs.items():
        df.to_csv(f'clusters csv\\{cluster}.csv')
