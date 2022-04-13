import pandas as pd
pd.set_option('display.max_columns', None)

def load_data():
    df_train = pd.read_csv('tf_mini.csv')
    return df_train

df_train = load_data()
print(df_train.head())
