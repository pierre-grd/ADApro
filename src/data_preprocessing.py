import pandas as pd
pd.set_option('display.max_columns', None)


def load_data(training_path, label_path):
    training_df = pd.read_csv(training_path)
    label_train = pd.read_csv(label_path)
    training_df.columns = training_df.columns.str.replace('track_id_clean', 'track_id')
    df = pd.merge(training_df, label_train, on='track_id')

    ## non-premium users can't skip tracks : filter premium == True
    df = df[df['premium'] == True]
    return df


def dataset_info(df):
    print(df.sample(3))
    print(df.describe())
    print('\n')
    print(
        f'This partitioned dataset consists of {df["session_id"].unique().size} sessions of premium users, who listened to or skipped  {df.shape[0]} tracks')

def dummy_creation(df):
    df["skipped"] = df["not_skipped"].apply(lambda x: 1 if x == False else 0)
    df = df.drop(["track_id", "not_skipped", "date", "release_year"], axis=1)
    return df

