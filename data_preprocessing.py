import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option('display.max_columns', None)


def load_data():
    training_df = pd.read_csv('data/log_mini.csv')
    label_train = pd.read_csv('data/tf_mini.csv')
    training_df.columns = training_df.columns.str.replace('track_id_clean', 'track_id')
    df = pd.merge(training_df, label_train, on='track_id')

    ## non-premium users can't skip tracks : filter premium == True
    df = df[df['premium'] == True]
    return df


def dataset_info():
    print(df.sample(3))
    print(df.describe())
    print(
        f'This partitioned dataset consists of {df["session_id"].unique().size} sessions of premium users, who listened to or skipped  {df.shape[0]} tracks')


def skip_matrix():
    df_skip = df[['skip_1', 'skip_2', 'skip_3', 'not_skipped']]
    corr = df_skip.corr(method='pearson')
    plt.figure(figsize=(8, 7))
    sns.heatmap(corr,
                xticklabels=corr.columns.values,
                yticklabels=corr.columns.values,
                )


def histogram_float_features():
    column = list(df.loc[:, df.dtypes == float].columns)
    df.hist(column=column)


df = load_data()
dataset_info()
skip_matrix()
histogram_float_features()
plt.show()
