from src.data_preprocessing import *
from src.EDA import *
from src.models import *
from src.NeuNets import *

training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

#====================
#data load
#====================

df = load_data(training_path, label_path)

# small dataset summary
dataset_info(df)



#====================
#EDA
#====================
skip_nonskip_distribution(df, "raw")

acoust = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
         'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']
matrix(df[acoust])

hist_continuous(df)

int_column = list(df.loc[:, df.dtypes == int].columns)
float_column = list(df.loc[:, df.dtypes == float].columns)
scatterplot_skip(df, int_column, float_column)

categorical_col = ['session_position', 'session_length', 'context_switch',
                  'no_pause_before_play', 'short_pause_before_play',
                  'long_pause_before_play', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback',
                  'premium', 'context_type',
                  'mode']

countplot(df, categorical_col)

#we choose to not downsample the data, since we're loosing accuracy if so
skip_nonskip_distribution(downsample(df), "downsample")
#====================
#Cleaning
#====================

df = dummy_creation(df)
df = normalize_float(df)



#=============================
#Model -> only track features
#=============================

X_train, X_test, y_train, y_test = split_data(df, only_track= True)
print(logistic_model(X_train, X_test, y_train, y_test))
print(GBC_model(X_train, X_test, y_train, y_test, hyper_tuning = False))
print(RF_model(X_train, X_test, y_train, y_test, hyper_tuning = False))

del X_train, X_test, y_train, y_test

#=============================
#Model -> all features
#=============================

X_train, X_test, y_train, y_test = split_data(df, only_track= False)
print(logistic_model(X_train, X_test, y_train, y_test))
print(GBC_model(X_train, X_test, y_train, y_test, hyper_tuning = False))
print(RF_model(X_train, X_test, y_train, y_test, hyper_tuning = False))

X_train, X_test, y_train, y_test = rnn_preprocess(X_train, X_test, y_train, y_test)
model = rnn_model(X_train)
rnn_train(model, X_train, X_test, y_train, y_test)

