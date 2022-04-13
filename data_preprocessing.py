import pandas as pd
pd.set_option('display.max_columns', None)

def load_data():
    training_df = pd.read_csv('log_mini.csv')
    label_train = pd.read_csv('tf_mini.csv')
    return label_train, training_df

df_train, training_df = load_data()

