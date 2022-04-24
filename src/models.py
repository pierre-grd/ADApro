from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report




def split_data(df):
    X_df = df[['duration', 'us_popularity_estimate', 'acousticness', 'beat_strength', 'bounciness', 'danceability',
               'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
               'liveness', 'loudness', 'mechanism', 'tempo', 'organism', 'speechiness', 'tempo', 'time_signature',
               'valence', 'acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
               'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']]
    y_df = df["skipped"]
    print(X_df)
    print(y_df)
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test


def GBC_model(X_train, X_test, y_train, y_test):
    GBC_model = GradientBoostingClassifier()
    GBC_model.fit(X_train, y_train)
    GBC_pred = GBC_model.predict(X_test)
    print(GBC_pred)
    print(y_test)
    return classification_report(y_test, GBC_pred)
