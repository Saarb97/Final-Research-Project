import pandas as pd
from scipy.stats import mannwhitneyu


def load_data(file_path):
    return pd.read_csv(file_path)


def group_data_by_cluster(data_df):
    return data_df.groupby('cluster')


def get_passed_prompts(data_df):
    return data_df[(data_df['performance'] == 1)].drop(['cluster', 'performance', 'text', 'named_entities'], axis=1)


def get_failed_prompts(data_df):
    return data_df[data_df['performance'] == 0].drop(['cluster', 'performance', 'text', 'named_entities'], axis=1)


def calculate_descriptive_stats(group):
    cluster_df = pd.DataFrame()
    cluster_df = cluster_df.assign(mean=group.drop(['text', 'cluster', 'performance', 'named_entities'], axis=1).mean(),
                                   median=group.drop(['text', 'cluster', 'performance', 'named_entities'], axis=1).median(),
                                   std=group.drop(['text', 'cluster', 'performance', 'named_entities'], axis=1).std(),
                                   min=group.drop(['text', 'cluster', 'performance', 'named_entities'], axis=1).min(),
                                   max=group.drop(['text', 'cluster', 'performance', 'named_entities'], axis=1).max())
    return cluster_df


def calculate_statistical_significance(group, passed_prompts):
    u_stats = []
    p_vals = []
    for col in group.drop(['text', 'cluster', 'performance', 'named_entities'], axis=1).columns:
        u_stat, p_val = mannwhitneyu(group[col], passed_prompts[col], alternative='two-sided')
        u_stats.append(u_stat)
        p_vals.append(p_val)
    return u_stats, p_vals


def process_cluster(group, passed_prompts):
    cluster_df = calculate_descriptive_stats(group)
    u_stats, p_vals = calculate_statistical_significance(group, passed_prompts)
    cluster_df = cluster_df.assign(u_stat=u_stats, p_val=p_vals)
    return cluster_df


if __name__ == '__main__':
    FILE_PATH = "full_dataset_feature_extraction_09-05.csv"
    data_df = load_data(FILE_PATH)
    cluster_groups = group_data_by_cluster(data_df)
    passed_prompts = get_passed_prompts(data_df)
    cluster_dfs = {}

    # Split the original dataset into individual CSV files by clusters
    for cluster, group in cluster_groups:
        group.to_csv(f'clusters csv\\{cluster}_data.csv', index=False)

    for cluster, group in cluster_groups:
        cluster_dfs[cluster] = process_cluster(group, passed_prompts)

    for cluster, df in cluster_dfs.items():
        df.to_csv(f'clusters csv\\{cluster}_statistics.csv')
