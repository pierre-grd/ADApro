from src.data_preprocessing import *
from src.viz import *

training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

# data load + cleaning
df = normalize_float(dummy_creation(load_data(training_path, label_path)))
# small dataset summary
dataset_info(df)

# EDA

acoust = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
          'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']
matrix(df[acoust])

hist_continuous(df)

scatterplot_skip(df, "energy", "duration")

categorical_col = ['session_position', 'session_length', 'context_switch',
                   'no_pause_before_play', 'short_pause_before_play',
                   'long_pause_before_play', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback',
                   'premium', 'context_type',
                   'mode']

countplot(df, categorical_col)
