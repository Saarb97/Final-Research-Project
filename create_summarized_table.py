import pandas as pd
from scipy.stats import mannwhitneyu


def load_data(file_path):
    return pd.read_csv(file_path)


def group_data_by_cluster(data_df):
    return data_df.groupby('cluster')


def get_passed_prompts(data_df, textual_cols):
                            # Target column
    return data_df[(data_df[textual_cols[1]] == 1)].drop(textual_cols, axis=1)


def get_failed_prompts(data_df, textual_cols):
                            # Target column
    return data_df[data_df[textual_cols[1]] == 0].drop(textual_cols, axis=1)


def calculate_descriptive_stats(group, textual_cols):
    cluster_df = pd.DataFrame()
    cluster_df = cluster_df.assign(mean=group.drop(textual_cols, axis=1).mean(),
                                   median=group.drop(textual_cols, axis=1).median(),
                                   std=group.drop(textual_cols, axis=1).std(),
                                   min=group.drop(textual_cols, axis=1).min(),
                                   max=group.drop(textual_cols, axis=1).max())
    return cluster_df


def calculate_statistical_significance(group, passed_prompts,  textual_cols):
    u_stats = []
    p_vals = []
    for col in group.drop(textual_cols, axis=1).columns:
        u_stat, p_val = mannwhitneyu(group[col], passed_prompts[col], alternative='two-sided')
        u_stats.append(u_stat)
        p_vals.append(p_val)
    return u_stats, p_vals


def process_cluster(group, passed_prompts, textual_cols):
    cluster_df = calculate_descriptive_stats(group, textual_cols)
    u_stats, p_vals = calculate_statistical_significance(group, passed_prompts, textual_cols)
    cluster_df = cluster_df.assign(u_stat=u_stats, p_val=p_vals)
    return cluster_df

def dataframe_to_string(df):
    # Convert DataFrame to a string in the desired format
    df_string = f"pd.DataFrame({{{', '.join([f'{col}: {df[col].tolist()}' for col in df.columns])}}}, index={df.index.tolist()})"
    return df_string

def create_cluster_summary_csv(cluster_groups, output_file, textual_cols):
    cluster_summary = []

    for cluster, group_df in cluster_groups:
        # Select only the "text" column and reset index for line numbers
        cluster_text_df = group_df[[textual_cols[0]]].reset_index()
        # Convert the DataFrame to a string representation
        cluster_text_str = dataframe_to_string(cluster_text_df)
        # Append cluster number and DataFrame string to the summary list
        cluster_summary.append([cluster, cluster_text_str])

    # Create a summary DataFrame
    summary_df = pd.DataFrame(cluster_summary, columns=['cluster', 'dataframe'])
    # Write the summary DataFrame to a CSV file
    summary_df.to_csv(output_file, index=False)


def _created_statistics_tables(num_clusters, textual_cols, destination):
    for i in range(num_clusters):  # Loop from 0_data.csv to (num_clusters-1)_data.csv
        file_name = f'{destination}\\{i}_data.csv'
        data_df = load_data(file_name)
        cluster_groups = group_data_by_cluster(data_df)
        passed_prompts = get_passed_prompts(data_df, textual_cols)
        cluster_dfs = {}

        for cluster, group in cluster_groups:
            cluster_dfs[cluster] = process_cluster(group, passed_prompts, textual_cols)
            cluster_dfs[cluster].to_csv(f'{destination}\\{cluster}_statistics.csv')

        # for cluster, df in cluster_dfs.items():
        #     df.to_csv(f'{destination}\\{cluster}_statistics.csv')

def _split_clusters_data(clusters_df, destination):
    '''
        TODO: Make it work with Linux as well
    '''
    cluster_groups = group_data_by_cluster(clusters_df)
    print(len(cluster_groups))
    for cluster, group in cluster_groups:
        group.to_csv(f'{destination}\\{cluster}_data.csv', index=False)

    return len(cluster_groups)

def create_summarized_tables(clusters_df, text_col_name, target_col_name,  destination):
    '''
        TODO: Make it work with Linux as well
    '''
    # Textual columns are emitted from statistical analysis of their contents.
    textual_cols = [text_col_name, target_col_name, 'cluster', 'named_entities']

    num_clusters = _split_clusters_data(clusters_df, destination)
    _created_statistics_tables(num_clusters, textual_cols, destination)

if __name__ == '__main__':
    '''
        Local test case
    '''
    FILE_PATH = "full_dataset_feature_extraction_09-05.csv"
    clusters_df = load_data(FILE_PATH)
    num_clusters = 20
    text_col_name = 'text'
    target_col_name = 'performance'
    destination = 'testfolder'
    create_summarized_tables(clusters_df, text_col_name, target_col_name, destination)


    # FILE_PATH = "full_dataset_feature_extraction_09-05.csv"
    # data_df = load_data(FILE_PATH)
    # cluster_groups = group_data_by_cluster(data_df)
    # passed_prompts = get_passed_prompts(data_df)
    # cluster_dfs = {}
    #
    # split_clusters = False
    # if split_clusters:
    #     for cluster, group in cluster_groups:
    #         group.to_csv(f'clusters csv\\{cluster}_data.csv', index=False)
    #
    # gen_statistics = True
    # if gen_statistics:
    #     for i in range(20):  # Loop from 0_data.csv to 19_data.csv
    #         file_name = f'clusters csv\\{i}_data.csv'
    #         data_df = load_data(FILE_PATH)
    #         cluster_groups = group_data_by_cluster(data_df)
    #         passed_prompts = get_passed_prompts(data_df)
    #         cluster_dfs = {}
    #
    #         for cluster, group in cluster_groups:
    #             cluster_dfs[cluster] = process_cluster(group, passed_prompts)
    #
    #         for cluster, df in cluster_dfs.items():
    #             df.to_csv(f'clusters csv\\{cluster}_statistics.csv')
