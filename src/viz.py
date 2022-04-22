import matplotlib.pyplot as plt
import seaborn as sns

def skip_matrix(df):
    df_skip = df[['skip_1', 'skip_2', 'skip_3', 'not_skipped']]
    corr = df_skip.corr(method='pearson')
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )


def histogram_float_features(df):
    column = list(df.loc[:, df.dtypes == float].columns)
    df.hist(column=column)