from src.data_preprocessing import *
from src.eda import *
from src.models import *
from src.neunets import *
import os

training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

# ====================
# data load
# ====================

df = load_data(training_path, label_path)

# small dataset summary
dataset_info(df)

# ====================
# EDA
# ====================
skip_nonskip_distribution(df, "raw", save_plot= True)

acoust = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
          'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']
matrix(df[acoust], save_plot= True)

hist_continuous(df, save_plot= True)

int_column = list(df.loc[:, df.dtypes == int].columns)
float_column = list(df.loc[:, df.dtypes == float].columns)
scatterplot_skip(df, int_column, float_column, save_plot= True)

categorical_col = ['session_position', 'session_length', 'context_switch',
                   'no_pause_before_play', 'short_pause_before_play',
                   'long_pause_before_play', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback',
                   'premium', 'context_type',
                   'mode']

countplot(df, categorical_col, save_plot= True)

# we choose to not downsample the data, since we're loosing accuracy if so
skip_nonskip_distribution(downsample(df), "downsample", save_plot= True)
# ====================
# Cleaning
# ====================

df = dummy_creation(df)
df = normalize_float(df)

# =============================
# Model -> only track features
# =============================

x_train, x_test, y_train, y_test = split_data(df, only_track=True)
#logit_report = train_logistic_model(x_train, x_test, y_train, y_test)
#plot_classification_report(logit_report)

#bgc_report = train_gbc_model(x_train, x_test, y_train, y_test, hyper_tuning=False)
#plot_classification_report(bgc_report)

rf_report = train_RF_model(x_train, x_test, y_train, y_test, hyper_tuning=False)
plot_classification_report(rf_report, save_plot= True)
del x_train, x_test, y_train, y_test

# =============================
# Model -> all features
# =============================
print("splitting data")
x_train, x_test, y_train, y_test = split_data(df, only_track=False)
print("train logit model")
logit_report = train_logistic_model(x_train, x_test, y_train, y_test)
plot_classification_report(logit_report, 'logit', save_plot= True)
print("train gbc model")
gbc_report = train_gbc_model(x_train, x_test, y_train, y_test, hyper_tuning=False)
plot_classification_report(gbc_report, 'gbc', save_plot= True)
print("train rf model")
rf_report = train_RF_model(x_train, x_test, y_train, y_test, hyper_tuning=False)
plot_classification_report(rf_report, 'rf', save_plot= True)


x_train, x_test, y_train, y_test = rnn_preprocess(x_train, x_test, y_train, y_test)
model = rnn_model(x_train)
rnn_train(model, x_train, x_test, y_train, y_test)

del x_train, x_test, y_train, y_test



script = """
tensorboard --logdir models/logs
"""
os.system("bash -c '%s'" % script)
#Then click on the http://localhost to analyse the training process and the performance of the model

