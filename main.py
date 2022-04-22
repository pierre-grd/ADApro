from src.data_preprocessing import *
from src.viz import *


training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

df = load_data(training_path, label_path)
dummy_creation(df)
dataset_info(df)
skip_matrix(df)
histogram_float_features(df)
plt.show()
