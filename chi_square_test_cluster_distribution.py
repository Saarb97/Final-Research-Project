import pandas as pd
from scipy.stats import chi2_contingency

if __name__ == '__main__':
    FILE_PATH = "full_dataset_feature_extraction_10-04.csv"
    df = pd.read_csv(FILE_PATH)
    # Step 1: Count the number of incorrect observations in each cluster
    incorrect_counts = df[df['performance'] == 0].groupby('cluster')['performance'].count()

    # Step 2: Calculate the total number of observations in each cluster
    total_counts = df.groupby('cluster')['performance'].count()

    # Step 3: Calculate the proportion of incorrect observations in each cluster
    incorrect_proportions = incorrect_counts / total_counts

    # Print the proportions for inspection
    print("Proportions of incorrect observations in each cluster:")
    print(incorrect_proportions)

    # Step 4: Calculate the percentage of incorrect observations in each cluster from the total amount of incorrect observations
    total_incorrect = incorrect_counts.sum()
    incorrect_percentage = (incorrect_counts / total_incorrect) * 100

    # Print the percentages for inspection
    print("\nPercentage of incorrect observations in each cluster from the total amount of incorrect observations:")
    print(incorrect_percentage)

    # Step 5: Conduct a chi-squared test to see if the distribution of incorrect observations is different across clusters
    contingency_table = pd.crosstab(df['cluster'], df['performance'])
    chi2, p, dof, expected = chi2_contingency(contingency_table)

    print("\nChi-squared test results:")
    print("Chi-squared statistic:", chi2)
    print("p-value:", p)

    # Interpret the results
    if p < 0.05:
        print("There is a significant difference in the distribution of incorrect observations across clusters.")
    else:
        print("There is no significant difference in the distribution of incorrect observations across clusters.")

    # Step 6: Conduct the second chi-squared test on incorrect_percentage
    # Convert the percentages back to counts for the chi-squared test
    expected_incorrect_counts = total_incorrect / len(incorrect_counts)  # Assuming equal distribution
    expected_percentage_counts = [expected_incorrect_counts] * len(incorrect_counts)

    chi2_2, p_2, dof_2, expected_2 = chi2_contingency([incorrect_counts, expected_percentage_counts])

    print("\nSecond Chi-squared test results (on incorrect_percentage):")
    print("Chi-squared statistic:", chi2_2)
    print("p-value:", p_2)

    # Interpret the results
    if p_2 < 0.05:
        print(
            "There is a significant difference in the percentage distribution of incorrect observations across clusters.")
    else:
        print(
            "There is no significant difference in the percentage distribution of incorrect observations across clusters.")