from src.data_preprocessing import *
from src.viz import *
from src.models import *


training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

#====================
#data load
#====================

df = load_data(training_path, label_path)

# small dataset summary
#dataset_info(df)



#====================
#EDA
#====================
#skip_nonskip_distribution(df)

#acoust = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
#         'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']
#matrix(df[acoust])

#hist_continuous(df)

#scatterplot_skip(df, "energy", "duration")

#categorical_col = ['session_position', 'session_length', 'context_switch',
#                  'no_pause_before_play', 'short_pause_before_play',
#                  'long_pause_before_play', 'hist_user_behavior_n_seekfwd', 'hist_user_behavior_n_seekback',
#                  'premium', 'context_type',
#                  'mode']

#countplot(df, categorical_col)

#====================
#Cleaning
#====================

df = dummy_creation(df)
df = normalize_float(df)
df = downsample(df)

#we choose to not downsample the data, since we're loosing accuracy if so
#skip_nonskip_distribution(df)

#====================
#Model
#====================

X_train, X_test, y_train, y_test = split_data(df, True)
print(GBC_model(X_train, X_test, y_train, y_test))
#print(RF_model(X_train, X_test, y_train, y_test))