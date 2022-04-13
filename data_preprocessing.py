import pandas as pd
pd.set_option('display.max_columns', None)

def load_data():
    training_df = pd.read_csv('log_mini.csv')
    label_train = pd.read_csv('tf_mini.csv')
    training_df.columns = training_df.columns.str.replace('track_id_clean', 'track_id')
    df = pd.merge(training_df, label_train, on='track_id')
    return df

df = load_data()
print(df.head())