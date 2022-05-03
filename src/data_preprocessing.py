import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

pd.set_option('display.max_columns', None)


def load_data(training_path, label_path):
    """
    :param training_path: A string that stores the data set path of the user's listening sessions information
    :param label_path: A string that stores the path of the dataset of information about the track's intrinsic characteristics
    :return:A pandas array with all the information (on the music itself and on the user sessions)
    """

    training_df = pd.read_csv(training_path)
    label_train = pd.read_csv(label_path)
    training_df.columns = training_df.columns.str.replace('track_id_clean', 'track_id')
    df = pd.merge(training_df, label_train, on='track_id')
    df["skipped"] = df["not_skipped"].apply(lambda x: 1 if x == False else 0)
    df = df.drop(["track_id", "not_skipped", "date", "release_year", "skip_1", "skip_2", "skip_3"], axis=1)
    del training_df, label_train
    return df


def dataset_info(df):
    print(df.sample(3))
    print('\n')
    print(df.describe())
    print('\n')


def dummy_creation(df):
    """
    :param df:
    :return:
    """

    df['session_id'] = pd.factorize(df['session_id'])[0]
    df = pd.get_dummies(df, columns=['hist_user_behavior_reason_start', 'hist_user_behavior_reason_end',
                                     'hist_user_behavior_is_shuffle', 'mode', 'premium', 'context_type',
                                     ], drop_first=True)
    return df


def normalize_float(df):
    """
    :param df: The pandas array with all the information (on the music itself and on the user sessions)
    :return: The transformed dataset: float columns are now normalized
    """
    column = list(df.loc[:, df.dtypes == float].columns)
    x = df[column].values
    x_scaled = StandardScaler().fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column, index=df.index)
    df[column] = df_temp
    del column, df_temp, x, x_scaled
    return df


def downsample(df):
    """
    This function allows to resample (to the less represented class) to have a balanced dataset

    :param df: The pandas array with all the information (on the music itself and on the user sessions)
    :return: The pandas array with balanced classes
    """
    skip = df[df["skipped"] == 0]
    nonskip = df[df["skipped"] == 1]
    nonskip_downsample = resample(nonskip,
                                  replace=True,
                                  n_samples=len(skip),
                                  random_state=42)

    df = pd.concat([skip, nonskip_downsample])
    del skip, nonskip, nonskip_downsample
    return df
