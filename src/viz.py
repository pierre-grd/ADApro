import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

def matrix(df):
    corr = df.corr(method='pearson')
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )

def hist_continuous(df):
    column = list(df.loc[:, df.dtypes == float].columns)
    df.hist(column=column)

def scatterplot_skip(df, col_name1, col_name2):
    plt.figure(figsize=(10, 5))
    sns.scatterplot(x=col_name1, y=col_name2, hue="skipped", data=df)


def countplot(df, col1):
    crosstab = pd.crosstab(df[col1], df["skipped"])
    crosstab.div(crosstab.sum(1).astype(float), axis=0).plot(kind="bar", figsize=(20, 5))
    plt.show()