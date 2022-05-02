import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import classification_report


def split_data(df, only_track=True):
    if only_track == True:
        X_df = df[['session_id', 'duration', 'us_popularity_estimate', 'acousticness', 'beat_strength', 'bounciness',
                   'danceability',
                   'dyn_range_mean', 'energy', 'flatness', 'instrumentalness', 'key',
                   'liveness', 'loudness', 'mechanism', 'tempo', 'organism', 'speechiness', 'tempo', 'time_signature',
                   'valence', 'acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3',
                   'acoustic_vector_4', 'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']]
    else:
        X_df = df.drop("skipped", axis=1)

    y_df = df["skipped"]

    X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.25, random_state=0)
    return X_train, X_test, y_train, y_test

def logistic_model(X_train, X_test, y_train, y_test):
    logistic_model = LogisticRegression()
    logistic_model.fit(X_train, y_train)
    logistic_prediction = logistic_model.predict(X_test)
    return classification_report(y_test, logistic_prediction)


def GBC_model(X_train, X_test, y_train, y_test, hyper_tuning = False):
    GBC_model = GradientBoostingClassifier()
    n_estimators = [int(x) for x in np.linspace(start=100, stop=1100, num=3)]
    learning_rate = [0.01, 0.1, 0.5]
    max_features = ['auto', 'sqrt']
    min_samples_leaf = [1, 2]
    random_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'learning_rate': learning_rate,
                   'min_samples_leaf': min_samples_leaf}

    if hyper_tuning == True:
        GBC_model = RandomizedSearchCV(GBC_model, param_distributions=random_grid, n_jobs=-1, n_iter=25, cv=3, verbose=2)
    else:
        GBC_model = GradientBoostingClassifier()

    GBC_model.fit(X_train, y_train)
    GBC_pred = GBC_model.predict(X_test)
    return classification_report(y_test, GBC_pred)


def RF_model(X_train, X_test, y_train, y_test, hyper_tuning = False):
    RF_model = RandomForestClassifier()
    n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num=3)]
    max_depth = [int(x) for x in np.linspace(2, 20, num=3)]
    min_samples_leaf = [1, 2, 5]
    random_grid = {'n_estimators': n_estimators,
                   'max_depth': max_depth,
                   'min_samples_leaf': min_samples_leaf}
    if hyper_tuning == True:
        RF_model = RandomizedSearchCV(RF_model, param_distributions=random_grid, n_jobs=-1, n_iter=25, cv=3, verbose=2)
    else:
        RF_model = GradientBoostingClassifier()

    RF_model.fit(X_train, y_train)
    RF_model_pred = RF_model.predict(X_test)
    return classification_report(y_test, RF_model_pred)
