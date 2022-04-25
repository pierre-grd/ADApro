import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


def split_data(df):
    X_df = df[['session_id', 'duration', 'us_popularity_estimate', 'acousticness', 'beat_strength', 'bounciness', 'danceability',
               'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
               'liveness', 'loudness', 'mechanism', 'tempo', 'organism', 'speechiness', 'tempo', 'time_signature',
               'valence', 'acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
               'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']]
    y_df = df["skipped"]
    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test


def GBC_model(X_train, X_test, y_train, y_test):
    GBC_model = GradientBoostingClassifier()
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    learning_rate = [float(x) for x in np.linspace(start=0.01, stop=0.5, num=5)]
    max_features = ['auto', 'sqrt']
    max_depth = [int(x) for x in np.linspace(100, 1100, num=11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 5]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'learning_rate': learning_rate,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}

    #GBC_model = RandomizedSearchCV(GBC_model, param_distributions=random_grid, n_jobs=-1, n_iter=25, cv=2, verbose=5)
    GBC_model.fit(X_train, y_train)
    GBC_pred = GBC_model.predict(X_test)
    return classification_report(y_test, GBC_pred)

def RF_model(X_train, X_test, y_train, y_test):
    RF_model = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=10)]
    max_depth = [int(x) for x in np.linspace(2, 20, num=11)]
    min_samples_split = [2, 5, 10]
    min_samples_leaf = [1, 2, 5]
    bootstrap = [True, False]
    random_grid = {'n_estimators': n_estimators,
                   'bootstrap': bootstrap,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf}
    RF_model = RandomizedSearchCV(RF_model, param_distributions=random_grid, n_jobs=-1, n_iter=25, cv=2, verbose=5)

    RF_model.fit(X_train, y_train)
    RF_model_pred = RF_model.predict(X_test)
    return classification_report(y_test, RF_model_pred)
