from typing import Any

import pandas as pd
from sklearn.ensemble._iforest import _average_path_length
import multiprocessing
from functools import partial
from sklearn.ensemble import IsolationForest
from causallearn.search.FCMBased import lingam
from concurrent.futures import ThreadPoolExecutor
from sklearn.cluster import MiniBatchKMeans, MeanShift
from sklearn.metrics import silhouette_score
from sklearn import metrics
from scipy.spatial.distance import cdist
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.metrics import silhouette_score, make_scorer
from sklearn.decomposition import PCA
import os
from sklearn.preprocessing import LabelEncoder


def convert_text_to_numeric(df, column_name):
    """
    Convert a textual categorical column into numerical categories.

    Args:
        df (pd.DataFrame): The input DataFrame.
        column_name (str): The name of the column to convert.

    Returns:
        pd.DataFrame: A DataFrame with the column transformed into numerical categories.
    """
    # Ensure the column exists in the DataFrame
    if column_name not in df.columns:
        raise ValueError(f"Column '{column_name}' does not exist in the DataFrame.")

    # Use LabelEncoder to transform the column
    encoder = LabelEncoder()
    df[column_name] = encoder.fit_transform(df[column_name])

    return df, encoder.classes_


class MeanShiftEstimator(BaseEstimator, ClusterMixin):
    def __init__(self, bandwidth=1.0):
        self.bandwidth = bandwidth
        self.mean_shift = None

    def fit(self, X, y=None):
        self.mean_shift = MeanShift(bandwidth=self.bandwidth)
        self.mean_shift.fit(X)
        return self

    def predict(self, X):
        return self.mean_shift.predict(X)

    def score(self, X, y=None):
        labels = self.predict(X)
        if len(set(labels)) > 1:
            return silhouette_score(X, labels)
        else:
            return -1  # If all samples are assigned to the same cluster


def custom_silhouette_scorer(estimator, X):
    labels = estimator.predict(X)
    if len(set(labels)) > 1:
        return silhouette_score(X, labels)
    else:
        return -1  # If all samples are assigned to the same cluster


def find_best_bandwidth(data, indices):
    df_for_clustering = data.iloc[:, list(indices)]
    params = {'bandwidth': np.arange(0.1, 2.0, 0.1)}
    mean_shift_estimator = MeanShiftEstimator()
    grid_search = GridSearchCV(mean_shift_estimator, param_grid=params, cv=3, scoring=custom_silhouette_scorer)
    grid_search.fit(df_for_clustering)
    return grid_search.best_params_['bandwidth']


def diffi_score(forest, X, inlier_samples="auto"):
    pred = forest.predict(X)
    X_out = X[pred < 0]
    X_in = X[pred > 0]

    if inlier_samples == "all":
        k = X_in.shape[0]
    elif inlier_samples == "auto":
        k = X_out.shape[0]
    else:
        k = int(inlier_samples)
    if k < X_in.shape[0]:
        breakpoint()
        X_in = X_in.iloc[np.random.choice(X_in.shape[0], k, replace=False), :]

    return (_mean_cumulative_importance(forest, X_out) /
            _mean_cumulative_importance(forest, X_in))


def _mean_cumulative_importance(forest, X):
    '''
    Computes mean cumulative importance for every feature of given forest on dataset X
    '''

    f_importance = np.zeros(X.shape[1])
    f_count = np.zeros(X.shape[1])

    if forest._max_features == X.shape[1]:
        subsample_features = False
    else:
        subsample_features = True

    for tree, features in zip(forest.estimators_, forest.estimators_features_):
        X_subset = X[:, features] if subsample_features else X

        importance_t, count_t = _cumulative_ic(tree, X_subset)

        if subsample_features:
            f_importance[features] += importance_t
            f_count[features] += count_t
        else:
            f_importance += importance_t
            f_count += count_t

    return f_importance / f_count


def _cumulative_ic(tree, X):
    '''
    Computes importance and count for every feature of given tree on dataset X
    '''
    importance = np.zeros(X.shape[1])
    count = np.zeros(X.shape[1])

    node_indicator = tree.decision_path(X)
    node_loads = np.array(node_indicator.sum(axis=0)).reshape(-1)
    # depth is number of edges in path, same as number of nodes in path -1
    depth = np.array(node_indicator.sum(axis=1), dtype=float).reshape(-1) - 1
    # when the tree is pruned (i.e. more than one instance at the leaf)
    # we consider the average path length to adjust depthה
    leaves_index = tree.apply(X)
    depth += _average_path_length(node_loads[leaves_index])

    iic = _induced_imbalance_coeff(tree, X, node_loads)
    rows, cols = node_indicator.nonzero()
    for i, j in zip(rows, cols):
        f = tree.tree_.feature[j]
        # ignore leaf nodes
        if f < 0:
            continue
        count[f] += 1
        importance[f] += iic[j] / depth[i]

    return importance, count


def _induced_imbalance_coeff(tree, X, node_loads):
    '''
    Computes imbalance coefficient for every *node* of a tree on dataset X
    '''
    # epsilon as defined in the original paper
    _EPSILON = 1e-2
    iic = np.zeros_like(node_loads)
    for i in range(len(iic)):
        # ignore leaf nodes
        if tree.tree_.children_left[i] < 0:
            continue
        n_left = node_loads[tree.tree_.children_left[i]]
        n_right = node_loads[tree.tree_.children_right[i]]
        if n_left == 0 or n_right == 0:
            iic[i] = _EPSILON
            continue
        if n_left == 1 or n_right == 1:
            iic[i] = 1
            continue
        iic[i] = max(n_left, n_right) / node_loads[i]
    return iic


def get_support(data, feature_id, feature_val, cluster):
    """This function compute support for a given value
    """
    n_cluster_size = len(cluster)
    num = 0
    for j in range(n_cluster_size):
        if data[cluster[j], feature_id] == feature_val:
            num = num + 1
    return num


def similarity_instance_cluster(data, instance_id, cluster):
    """This function computes the similarity between a new instance
    data[instance_id] and a cluster specified by cluster_id, with parallel computation.

    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        matrix containing original data

    instance_id: int
        row number of the new instance

    cluster: list
        a list containing the ids of instances in this cluster

    Returns
    -------
    sim: float
        the similarity between the input instance and input cluster
    """
    n_instances, n_features = data.shape
    sim = 0.0

    with ThreadPoolExecutor() as executor:
        for i in range(n_features):
            # Use a set to store unique values for faster membership testing
            unique = set(data[cluster, i])

            # Parallelize the computation of support values
            future_to_value = {executor.submit(get_support, data, i, value, cluster): value for value in unique}
            temp = sum(future.result() for future in future_to_value.keys())

            # Calculate the similarity for the current feature
            if temp > 0:
                sim += get_support(data, i, data[instance_id, i], cluster) / temp

    return sim


def squeezer_parallel(data, thre):
    """This function implements squeezer algorithm base on the paper "Squezzer
    : An Efficient Algorithm for Clustering Categorical Data", with parallelization.

    Parameters
    ----------
    data: array, shape(n_instances,n_features)
        the original data that need to be clustered, note that we donnot have
        to specify the number of clusters here

    thre: threshold used to decide if creating a new cluster is necessary

    Returns
    -------
    label: list, length(n_instances)
        label for every instance, label is a list of lists,list[i] represents
        cluster i, list[i] is a list containing the instances ID of cluster i
    """
    # Initialize the clustering result
    label = [[0]]

    # Obtain the number of instances and features from input data
    n_instances, n_features = data.shape
    print(f'num of instances: {n_instances}')

    # Create a pool of workers
    pool = multiprocessing.Pool()

    for i in range(1, n_instances):
        print(f'instance {i}')
        # Current number of clusters
        n_clusters = len(label)

        # Compute similarity between data[i,:] and each cluster in parallel
        func_partial = partial(similarity_instance_cluster, data, i)
        sim = pool.map(func_partial, [label[j] for j in range(n_clusters)])

        sim_max = max(sim)
        sim_max_cluster_id = sim.index(sim_max)

        if sim_max >= thre:
            label[sim_max_cluster_id].append(i)
        else:
            label.append([i])

    # Close the pool of workers
    pool.close()
    pool.join()

    return label


def append_cluster_ids_and_save(data_df, labels, output_file_path):
    """
    Appends the cluster ID of each instance to the original DataFrame and saves it as a CSV file.

    Parameters
    ----------
    data_df : pandas.DataFrame
        The original DataFrame containing the instances.
    labels : list of lists
        The output of squeezer_parallel, where each sublist contains the indices of instances in a cluster.
    output_file_path : str
        The path to save the CSV file.
    """
    # Create a dictionary mapping instance index to cluster ID
    cluster_ids = {}
    for cluster_id, cluster in enumerate(labels):
        for instance_index in cluster:
            cluster_ids[instance_index] = cluster_id

    # Append the cluster ID to the original DataFrame
    data_df['Cluster_ID'] = data_df.index.map(cluster_ids)

    # Save the DataFrame as a CSV file
    data_df.to_csv(output_file_path, index=False)


def find_optimal_k(data, k_range):
    """
    Finds the optimal number of clusters (k) with the highest Silhouette score.

    Parameters:
    - data: The input data for clustering.
    - k_range: A range of values for k to be tested.

    Returns:
    - The value of k that resulted in the highest Silhouette score.
    """
    best_k = None
    best_score = -np.inf

    for k in k_range:
        print(f'k={k}')
        kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(data)
        if len(np.unique(labels)) > 1:  # Silhouette score is not defined for a single cluster
            score = silhouette_score(data, labels)
            if score > best_score:
                best_score = score
                best_k = k

    return best_k


def plot_elbow_method(data, k_range, output_file_path):
    """
    Plots the Elbow method graph and saves it as an image file.

    Parameters:
    - data: The input data for clustering.
    - k_range: A range of values for k to be tested.
    - output_file_path: The path where the plot image will be saved.
    """
    # distortions = []
    #
    # for k in k_range:
    #     kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
    #     kmeans.fit(data)
    #     distortions.append(kmeans.inertia_)
    #
    # plt.figure(figsize=(8, 4))
    # plt.plot(k_range, distortions, 'bx-')

    distortions = []
    inertias = []
    mapping1 = {}
    mapping2 = {}

    for k in k_range:
        # Building and fitting the model
        kmeanModel = MiniBatchKMeans(n_clusters=k, batch_size=100)
        kmeanModel.fit(data)

        distortions.append(sum(np.min(cdist(data, kmeanModel.cluster_centers_,
                                            'euclidean'), axis=1)) / data.shape[0])
        inertias.append(kmeanModel.inertia_)

        mapping1[k] = sum(np.min(cdist(data, kmeanModel.cluster_centers_,
                                       'euclidean'), axis=1)) / data.shape[0]
        mapping2[k] = kmeanModel.inertia_

    plt.figure(figsize=(8, 4))
    plt.plot(k_range, distortions, 'bx-')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.title('The Elbow Method showing the optimal k')
    plt.savefig(output_file_path)
    plt.close()


def get_optimal_k(df, max_k=100, output_file_path='elbow_method.png'):
    k_range = range(2, max_k)  # Adjust the range based on your dataset and needs
    plot_elbow_method(df, k_range, output_file_path)
    optimal_k = find_optimal_k(df, k_range)
    print(f"The optimal number of clusters is: {optimal_k}")


def read_data(file_path):
    print(f'Reading file: {file_path}')
    return pd.read_csv(file_path)


def run_icalingam(data):
    print('Starting ICALiNGAM')

    # Define the columns to drop before analysing results
    columns_to_drop = ['text', 'performance', 'named_entities', 'Unnamed: 0']

    # Filter columns that exist in the data
    existing_columns_to_drop = [col for col in columns_to_drop if col in data.columns]
    data.drop(columns=existing_columns_to_drop, inplace=True)
    df_selection = data


    columns = range(len(df_selection.columns))
    feature_importance = []
    for i in range((len(columns) // 100) + 1):
        model = lingam.ICALiNGAM(42, 3000)
        until = min(len(columns), (1 + (i + 1) * 100))
        model.fit(df_selection.iloc[:, [columns[0]] + list(columns[(1 + i * 100):until])])
        if len(feature_importance) != 0:
            feature_importance = np.concatenate((feature_importance, model.adjacency_matrix_[0][1:]), axis=0)
        else:
            feature_importance = model.adjacency_matrix_[0][1:]
    feature_importance = np.concatenate(([0], feature_importance), axis=0)
    indices = np.argwhere(feature_importance != 0).flatten()
    return indices


def run_icalingam_v2(data):
    print('Starting ICALiNGAM')
    df_selection = data.drop(columns=['text', 'performance'])
    columns = range(len(df_selection.columns))
    feature_importance = []
    for i in range((len(columns) // 100) + 1):
        model = lingam.ICALiNGAM(42, 3000)
        until = min(len(columns), (1 + (i + 1) * 100))
        model.fit(df_selection.iloc[:, [columns[0]] + list(columns[(1 + i * 100):until])])
        if len(feature_importance) != 0:
            feature_importance = np.concatenate((feature_importance, model.adjacency_matrix_[0][1:]), axis=0)
        else:
            feature_importance = model.adjacency_matrix_[0][1:]
    feature_importance = np.concatenate(([0], feature_importance), axis=0)
    indices = np.argwhere(feature_importance != 0).flatten()

    new_indices = []
    for indice in indices:
        print(f'calculating index: {indice} out of {len(indices)}')
        feature_importance = []
        for i in range((len(columns) // 100) + 1):
            model = lingam.ICALiNGAM(22, 1000)
            untill = min(len(columns), (1 + (i + 1) * 100))
            elem_list = list(columns[(1 + i * 100):untill])
            if indice in elem_list:
                elem_list.remove(indice)

            model.fit(df.iloc[:, [columns[indice]] + elem_list])
            if len(feature_importance) != 0:
                feature_importance = np.concatenate((feature_importance, model.adjacency_matrix_[0][1:]), axis=0)
            else:
                feature_importance = model.adjacency_matrix_[0][1:]
        feature_importance = np.concatenate(([0], feature_importance), axis=0)
        indecies_current = np.argwhere(feature_importance >= 0.2)
        indecies_current = list(indecies_current.flatten())

        # new_indices = [new_indices + indecies_current]
        new_indices.extend(indecies_current)
    new_indices = (list(set(new_indices)))
    return new_indices


def cluster_data(data, indices, bandwidth):
    df_for_clustering = data.iloc[:, list(indices)]
    # kmean_model = MiniBatchKMeans(n_clusters=20, batch_size=100).fit(df_for_clustering)
    # return kmean_model.predict(df_for_clustering)
    mean_shift_model = MeanShift(bandwidth=bandwidth).fit(df_for_clustering)
    # min_bin_freq
    return mean_shift_model.predict(df_for_clustering)


def visualize_clusters(data, predictions, file_name):
    pca = PCA(n_components=2)
    reduced_data = pca.fit_transform(data)
    plt.figure(figsize=(10, 7))
    scatter = plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=predictions, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('Cluster Visualization')
    plt.xlabel('PCA Component 1')
    plt.ylabel('PCA Component 2')
    plt.savefig(file_name)
    plt.close()


def get_feature_selection_indices(path):
    df = read_data(path)
    indices = run_icalingam(df)
    return indices


def get_clusters(path, indices):
    df = read_data(path)
    best_bandwidth = find_best_bandwidth(df, indices)
    predictions = cluster_data(df, indices, best_bandwidth)
    clustered_data_df = df[['text', 'performance']].copy()
    clustered_data_df.loc[:, 'cluster'] = predictions
    cluster_col = clustered_data_df.pop('cluster')
    clustered_data_df.insert(2, 'cluster', cluster_col)
    return clustered_data_df


def selection_and_clustering(path):
    indices = get_feature_selection_indices(path)
    post_selection_clustered_data_df = get_clusters(path, indices)
    return post_selection_clustered_data_df


def save_df_to_csv(df, file_name):
    df.to_csv(file_name, index=False)


if __name__ == '__main__':
    # FILE_PATH = "fairness_bbq_dataset_with_embeddings.csv"
    FILE_PATH = os.path.join('testfolder2', 'test_feature_extraction_file.csv')
    df = read_data(FILE_PATH)
    df, class_mapping = convert_text_to_numeric(df, 'category')
    print(df)
    print("Class Mapping:", class_mapping)
    df_copy = df.copy()
    indices = run_icalingam(df_copy)
    df_copy = df.copy()
    best_bandwidth = find_best_bandwidth(df_copy, indices)
    # indices = run_icalingam_v2(df)
    df_copy = df.copy()
    predictions = cluster_data(df_copy, indices, best_bandwidth)

    post_selection_df = df[['text', 'category']].copy()
    # post_selection_df = df[['text', 'performance']].copy()
    post_selection_df.loc[:, 'cluster'] = predictions
    cluster_col = post_selection_df.pop('cluster')
    post_selection_df.insert(2, 'cluster', cluster_col)

    save_df_to_csv(post_selection_df, 'twitter_clustering24_11_24.csv')
    df_for_clustering = df.iloc[:, list(indices)]
    visualize_clusters(df_for_clustering, predictions, 'cluster_visualization.png')
