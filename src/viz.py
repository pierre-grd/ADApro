import matplotlib.pyplot as plt
import seaborn as sns


def matrix(df):
    corr = df.corr(method='pearson')
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )
    plt.savefig('plots/' + 'corr_matrix' + '.png')

def skip_nonskip_distribution(df, name):
    sns.countplot(x="skipped", data=df)
    plt.savefig('plots/' + str(name) + '.png')

def hist_continuous(df):
    column = list(df.loc[:, df.dtypes == float].columns)
    df.hist(column=column, figsize=(20, 8))
    plt.savefig('plots/' + 'hist' + '.png')


def scatterplot_skip(df, col_name1, col_name2):
    plt.figure(figsize=(10, 5))
    for col_x in col_name1:
        for col_y in col_name2:
            sns.scatterplot(x=col_x, y=col_y, hue="skipped", data=df)
            plt.savefig('plots/' + str(col_x) + '_' + str(col_y) + '.png')


def countplot(df, column):
    for col in column:
        sns.countplot(data=df, x=col)
        plt.savefig('plots/' + str(col) + '.png')
