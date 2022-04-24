from src.data_preprocessing import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_curve,roc_auc_score,confusion_matrix,classification_report
#from xgboost import XGBClassifier

training_path = 'data/log_mini.csv'
label_path = 'data/tf_mini.csv'

# data load + cleaning
df = normalize_float(dummy_creation(load_data(training_path, label_path)))

X_df = df.drop("skipped", axis=1)
y_df = df["skipped"]

X_train, X_test, y_train, y_test = train_test_split(X_df, y_df, test_size=0.4, random_state=0)
print(X_train)
print(y_train)