import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import mlflow.sklearn
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

mlflow.set_tracking_uri('http://localhost:8000/')
mlflow.set_experiment('hw3_models')


def preprocessing(data):
    data = data.dropna()
    cat_features = data.select_dtypes(include=['object']).columns.tolist()
    if cat_features is not None:
        for col in cat_features:
            data.loc[:, col] = LabelEncoder().fit_transform(data.loc[:, col])
    return data


df_train = pd.read_csv('data/titanic_train.csv')
df_train = preprocessing(df_train)
df_test = pd.read_csv('data/titanic_test.csv')
df_test = preprocessing(df_test)
params = {
    'max_depth': 2,
    'random_state': 10}
features = ['PassengerId', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']
target = ['Survived']
X_train, X_test, y_train, y_test = train_test_split(df_train[features], df_train[target],
                                                    test_size=0.2, random_state=42)
model = RandomForestClassifier(**params)
name = '7_rfc_model'
version = 1
model = mlflow.pyfunc.load_model(model_uri=f'models:/{name}/{version}')
# Делаем предсказание
pred = model.predict(df_test)
print(pred)