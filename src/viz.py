import matplotlib.pyplot as plt
import seaborn as sns


def matrix(df):
    corr = df.corr(method='pearson')
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )
    plt.show()


def hist_continuous(df):
    column = list(df.loc[:, df.dtypes == float].columns)
    df.hist(column=column, figsize=(20, 8))
    plt.show()


def scatterplot_skip(df, col_name1, col_name2):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=col_name1, y=col_name2, hue="skipped", data=df)
    plt.show()


def countplot(df, column):
    for col in column:
        sns.countplot(data=df, x=col)
        plt.show()
