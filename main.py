from src.data_preprocessing import *
from src.viz import *

training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

# data load + cleaning
df = dummy_creation(load_data(training_path, label_path))

# small infos in dataset
dataset_info(df)

# EDA
skip = ['skip_1', 'skip_2', 'skip_3']
matrix(df[skip])

acoust = ['acoustic_vector_0', 'acoustic_vector_1', 'acoustic_vector_2', 'acoustic_vector_3', 'acoustic_vector_4',
          'acoustic_vector_5', 'acoustic_vector_6', 'acoustic_vector_7']
matrix(df[acoust])

hist_continuous(df)
scatterplot_skip(df, "energy", "duration")

count_col = ["session_position", "context_type", "hist_user_behavior_is_shuffle"]
for col in count_col:
    countplot(df, col)

plt.show()
