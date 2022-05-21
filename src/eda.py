import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def matrix(df, save_plot = False):
    corr = df.corr(method='pearson')
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )

    if save_plot:
        plt.savefig('plots/' + 'corr_matrix' + '.png')
        plt.clf()
    else:
        plt.show()

def skip_nonskip_distribution(df, name, save_plot = False):
    sns.countplot(x="skipped", data=df)

    if save_plot:
        plt.savefig('plots/' + str(name) + '.png')
        plt.clf()
        plt.close()
    else:
        plt.show()


def hist_continuous(df, save_plot = False):
    column = list(df.loc[:, df.dtypes == float].columns)
    df.hist(column=column, figsize=(20, 8))

    if save_plot:
        plt.savefig('plots/' + 'hist' + '.png')
        plt.clf()
        plt.close()
    else:
        plt.show()


def scatterplot_skip(df, col_name1, col_name2, save_plot = False):
    plt.figure(figsize=(10, 5))
    for col_x in col_name1:
        for col_y in col_name2:
            sns.scatterplot(x=col_x, y=col_y, hue="skipped", data=df)
            if save_plot:
                plt.savefig('plots/' + str(col_x) + '_' + str(col_y) + '.png')
                plt.clf()
                plt.close()
            else:
                plt.show()


def countplot(df, column, save_plot = False):
    for col in column:
        sns.countplot(data=df, x=col)
        if save_plot:
            plt.savefig('plots/' + str(col) + '.png')
            plt.clf()
            plt.close()
        else:
            plt.show()

def plot_classification_report(clf_report, model_name :str, save_plot = False):
    clf_report = pd.DataFrame(clf_report).iloc[:-1, :].T
    sns.heatmap(clf_report, annot=True)
    if save_plot:
        plt.savefig('plots/' + str(model_name) + '.png')
        plt.clf()
        plt.close()
    else:
        plt.show()