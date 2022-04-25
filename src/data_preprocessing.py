import pandas as pd
from sklearn import preprocessing
from sklearn.utils import resample

pd.set_option('display.max_columns', None)


def load_data(training_path, label_path):
    training_df = pd.read_csv(training_path) # for now nrow=5000 -> easier to compute & analyze if problems
    label_train = pd.read_csv(label_path)
    training_df.columns = training_df.columns.str.replace('track_id_clean', 'track_id')
    df = pd.merge(training_df, label_train, on='track_id')
    del training_df, label_train
    return df


def dataset_info(df):
    print(df.sample(3))
    print(df.describe())
    print('\n')


def dummy_creation(df):
    df["skipped"] = df["not_skipped"].apply(lambda x: 1 if x == False else 0)
    df = df.drop(["track_id", "not_skipped", "date", "release_year" ], axis=1)
    df['session_id'] = pd.factorize(df['session_id'])[0]
    return df


def normalize_float(df):
    column = list(df.loc[:, df.dtypes == float].columns)
    x = df[column].values
    x_scaled = preprocessing.MinMaxScaler(feature_range=(0, 1)).fit_transform(x)
    df_temp = pd.DataFrame(x_scaled, columns=column, index=df.index)
    df[column] = df_temp
    del column, df_temp, x, x_scaled
    return df

def downsample(df):
    skip = df[df["skipped"] == 0]
    nonskip = df[df["skipped"] == 1]
    nonskip_downsample = resample(nonskip,
                              replace=True,
                              n_samples=len(skip),
                              random_state=42)


    df = pd.concat([skip, nonskip_downsample])
    del skip, nonskip, nonskip_downsample
    return df