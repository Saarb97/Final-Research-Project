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


def get_optimal_k(df,max_k=100, output_file_path='elbow_method.png'):
    k_range = range(2, max_k)  # Adjust the range based on your dataset and needs
    plot_elbow_method(df, k_range, output_file_path)
    optimal_k = find_optimal_k(df, k_range)
    print(f"The optimal number of clusters is: {optimal_k}")


if __name__ == '__main__':
    #FILE_PATH = "fairness_bbq_dataset_with_embeddings.csv"
    FILE_PATH = "full_dataset_feature_extraction_12-03-24.csv"
    print(f'reading file: {FILE_PATH}')
    orig_df = pd.read_csv(FILE_PATH)
    # y = orig_df.performance
    # X = df.drop(columns=['text', 'performance'])

    # df = pd.read_csv('pii_financial_dataset_with_embeddings.csv')
    # y = df.Performance
    # X = df.drop(columns=['Target', 'Performance', 'index'])

    # print(f'creating isolation forest')
    # clf = IsolationForest()
    # clf.fit(X)
    # feature_importance = diffi_score(clf, X)
    # print(f'creating indices')
    # indecies = np.argwhere(feature_importance > 10000)
    # indecies = list(indecies.flatten())

    # df = df.drop(columns=['text'])
    # df = df.drop(columns=['Target', 'index'])
    df_selection = orig_df
    #df_selection = df_selection.drop(columns=['text', 'performance'])
    df_selection = df_selection.drop(columns=['text', 'performance', 'cluster'])
    columns = range(0, len(df_selection.columns))

    feature_importance = []
    print(f'starting ICALiNGAM')
    for i in range((len(columns) // 100) + 1):
        model = lingam.ICALiNGAM(42, 1000)
        untill = min(len(columns), (1 + (i + 1) * 100))
        ling = model.fit(df_selection.iloc[:, [columns[0]] + list(columns[(1 + i * 100):untill])])
        if len(feature_importance) != 0:
            feature_importance = np.concatenate((feature_importance,
                                                 model.adjacency_matrix_[0][1:]), axis=0)
        else:
            feature_importance = model.adjacency_matrix_[0][1:]

    feature_importance = np.concatenate(([0], feature_importance), axis=0)
    indices = np.argwhere(feature_importance != 0)
    indices = list(indices.flatten())

    # new df
    # df_isolation = orig_df.iloc[:, list(indices)]i.join(org_df[['performance']])
    df_for_clustering = orig_df.iloc[:, list(indices)]
    df_for_clustering.to_csv('feature_selection_test_070424.csv')
    df_for_clustering = df_for_clustering[df_for_clustering['performance'] == 0]
    df_for_clustering = df_for_clustering.drop(columns=['performance'])
    print(df_for_clustering)
    n_instances, _ = df_for_clustering.shape
    # optimal_k = get_optimal_k(max_k=100,output_file_path='elbow_method.png',df=df_isolation)
    kmeanModel = MiniBatchKMeans(n_clusters=191, batch_size=100).fit(df_for_clustering)
    prediction = kmeanModel.predict(df_for_clustering)

    orig_df['cluster'] = -1  # Initialize all clusters to -1
    orig_df.loc[orig_df['performance'] == 0, 'cluster'] = prediction
    orig_df = orig_df.iloc[:, list(indices)].join(orig_df[['text','cluster']])
    #orig_df.to_csv('bad_prompts_clustering.csv')
    print(orig_df)
