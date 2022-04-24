import pandas as pd
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
pd.set_option('display.max_columns', None)


def load_data(training_path, label_path):
    training_df = pd.read_csv(training_path)
    label_train = pd.read_csv(label_path)
    training_df.columns = training_df.columns.str.replace('track_id_clean', 'track_id')
    df = pd.merge(training_df, label_train, on='track_id')
    return df


def dataset_info(df):
    print(df.sample(3))
    print(df.describe())
    print('\n')
    print(
        f'This partitioned dataset consists of {df["session_id"].unique().size} sessions of premium users, who listened to or skipped  {df.shape[0]} tracks')


def dummy_creation(df):
    df = pd.get_dummies(df, columns=['hist_user_behavior_reason_start', 'hist_user_behavior_reason_end',
                                     'hist_user_behavior_is_shuffle', 'mode', 'premium', 'context_type', 'skip_1',
                                     'skip_2',
                                     'skip_3'], drop_first=True)
    df["skipped"] = df["not_skipped"].apply(lambda x: 1 if x == False else 0)
    df = df.drop(["track_id", "not_skipped", "date", "release_year"], axis=1)
    return df


def normalize_float(df):
    column = list(df.loc[:, df.dtypes == float].columns)
    x = df[column].values
    x_scaled = min_max_scaler.fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column, index=df.index)
    df[column] = df_temp
    return df
