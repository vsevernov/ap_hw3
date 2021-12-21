from sklearn.preprocessing import LabelEncoder
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
import pandas as pd
import joblib
import json
from flask import Flask
from flask_restx import Api
import sqlite3
import logging

import mlflow.sklearn
from mlflow.models.signature import infer_signature
import mlflow.pyfunc

logging.basicConfig(filename="flask_app_log.log", format='%(asctime)s - %(message)s', level=logging.INFO)
logging.info('Новый запуск приложения')

app = Flask(__name__)
api = Api(app)


class MLModels:
    def __init__(self):
        self.model_info = {}
        self.models_type = [{'name': 'RandomForestClassifier', 'task': 'classification'},
                            {'name': 'lightgbm', 'task': 'classification'},
                            {'name': 'DecisionTreeRegressor', 'task': 'regression'}]

    def preprocessing(self, data):
        """
        Функция для предобработки входных данных (удаление null значений и преобразование
        категориальных значений в числовые)
        :param data: данные, которые необходимо предобработать
        :return: предобработанные данные
        """
        logging.info("Получение данных и препроцессинг")
        data = data.dropna()
        logging.info("Успешно удалили null значения")
        cat_features = data.select_dtypes(include=['object']).columns.tolist()
        logging.info(f"категориальные значения {cat_features}")
        if cat_features is not None:
            for col in cat_features:
                data.loc[:, col] = LabelEncoder().fit_transform(data.loc[:, col])
        logging.info("Done!")
        return data

    def model_train(self, input_data_train):
        """
        В этой функции проходит обучение модели, сохранение обученных моделей в ./model и
        формирование отчета по обученным моделям в БД (flask_app.db).
        Структура отчета:
        (
        dataset: на каком датасете обучалась модель
        features: фичи на которых обучаласть модель
        target: целевая переменная
        metric: метрика качества
        model_name: тип модели (RandomForestClassifier, lightgbm,...)
        name: имя модели как ее назвал пользователь
        exist_models: список всех обученных моделей
        exist_models_type: доступные классы моделей
        models_description: характеристики всех обученных моделей
        )
        :param input_data_train: Данные для обучения (передаются пользователем по API)
        :return: Возвращает эксземпляр класса
        """
        logging.info("Стадия обучения")
        features = input_data_train['features']
        target = input_data_train['target']
        data = pd.read_json(input_data_train['data'])
        params = input_data_train['params']
        model_name = input_data_train['model_name']
        dataset = input_data_train['dataset']
        name = input_data_train['name']
        logging.info('Успешно получили данные для обучения и теста')

        # Вибираем тип модели, который планируется обучить
        logging.info('Выбираем тип модели')
        if model_name == 'RandomForestClassifier':
            metric_name = 'Accuracy'
            logging.info('Выбиранный тип модели - RandomForestClassifier')
            self.model = RandomForestClassifier(**params)
            metric = accuracy_score
        elif model_name == 'lightgbm':
            metric_name = 'Accuracy'
            logging.info('Выбиранный тип модели - lightgbm')
            self.model = lgb.LGBMClassifier(**params)
            metric = accuracy_score
        elif model_name == 'DecisionTreeRegressor':
            metric_name = 'MSE'
            self.model = DecisionTreeRegressor(**params)
            logging.info('Выбиранный тип модели - DecisionTreeRegressor')
            metric = mean_squared_error

        # Если модель есть в списке обученных моделей, то обучим ее еще раз
        logging.info(f'Имя модели - {name}')
        if f'{name}.pkl' in os.listdir('model'):
            self.model = joblib.load(f'model/{name}.pkl')

        mlflow.set_tracking_uri('http://localhost:8000/')
        mlflow.set_experiment('hw3_models')

        with mlflow.start_run(run_name=f'{name}'):
            # Проводим предобработку обучающего датасета
            train_df = self.preprocessing(data)
            # Разбиваем на train/test
            X_train, X_test, y_train, y_test = train_test_split(train_df[features], train_df[target],
                                                                test_size=0.2, random_state=42)
            # Обучаем модель
            self.model.fit(X_train, y_train.values.ravel())
            # Получаем качество модели в зависимости от класса модели
            _metric = metric(y_test, self.model.predict(X_test))
            print(_metric)
            mlflow.log_param("params", params)
            mlflow.log_metric("score", _metric)
            signature = infer_signature(X_test, self.model.predict(X_test))
            mlflow.sklearn.log_model(self.model, f'{name}', signature=signature, registered_model_name=f'{name}')

        logging.info(f'Качество модели на отложенной выборки - {metric_name} = {_metric}')

        # Собираем характеристики модели в словарь
        self.model_info['name'] = f'{name}.pkl'
        self.model_info['model_name'] = model_name
        self.model_info['metric'] = _metric
        self.model_info['dataset'] = dataset
        self.model_info['features'] = features
        self.model_info['target'] = target

        # Сохраняем обученную модель в ./model
        logging.info(f'Сохраняем модель как model/{name}.pkl')
        joblib.dump(self.model, f'model/{name}.pkl')

        # Записываем результат в БД
        logging.info(f'Записываем результаты в БД')
        db = sqlite3.connect('flask_app.db')
        sql = db.cursor()
        sql.execute("""
        CREATE TABLE IF NOT EXISTS vvs_models_info (
        name TEXT PRIMARY KEY,
        model_name TEXT,
        metric REAL,
        dataset TEXT,
        features TEXT,
        target TEXT)
        """)
        db.commit()

        logging.info(f'Извлекаем информацию об обученной модели из БД flask_app.db')
        sql.execute(f'SELECT name FROM vvs_models_info WHERE name = "{name}.pkl"')
        print(sql.fetchone())
        if sql.fetchone() is not None:
            message = 'file exist, try another name'
            db.close()
        else:
            sql.execute(f"INSERT INTO vvs_models_info VALUES (?, ?, ?, ?, ?, ?)",
                        (f'{name}.pkl', str(model_name), _metric, dataset, str(features), str(target)))
            db.commit()
            db.close()
            message = 'success'
        print(message)
        logging.info(f"Информания о модели {self.model_info}!")
        logging.info("Done!")

        return message

    def pretrained_model_info(self, name):
        """
        Извлекаем информацию о модели из БД (flask_app.db)
        :param name: имя модели, информацию о которой хотим получить
        :return: dict с информацией о модели
        """
        logging.info(f"Получаем информацию о моделе {name}")
        db = sqlite3.connect('flask_app.db')
        sql = db.cursor()
        keys = ['name', 'model_name', 'metric', 'dataset', 'features', 'target']
        values = [v for v in sql.execute(f'SELECT * FROM vvs_models_info WHERE name = "{name}.pkl"')]
        model_info = dict(zip(keys, values[0]))
        logging.info(f"Done")
        return model_info

    def available_models_info(self):
        """
        Получаем информацию обо всех ранее обученных моделях
        :return: словарь с необходимой информацией
        """
        logging.info(f"Получаем информацию обо всех обученных моделях")
        db = sqlite3.connect('flask_app.db')
        sql = db.cursor()
        models_info = [info for info in sql.execute(f'SELECT * FROM vvs_models_info')]
        db.close()
        return {'models_description': models_info}

    def available_сlass_model(self):
        """
        Возвращает доступные классы моделей
        :return:
        """
        logging.info(f"Получаем информацию о доступных классах моделей")
        return {'class of models': self.models_type}

    def predict_model_mlflow(self, input_data_predict, name, version):
        logging.info(f"Стадия предсказания")
        logging.info(f"Имя модели с помощью которой хотим получить предсказание {name}")
        try:
            logging.info(f"Считываем данные для предсказания")
            data = pd.read_json(input_data_predict['data'])
            # Предобрабатываем входные данные
            logging.info(f"Делаем предобработку")
            test_df = self.preprocessing(data)
            logging.info(f"Импортируем модель model/{name}.pkl")
            # Импорт модели, которую будем использовать
            model = mlflow.pyfunc.load_model(model_uri=f'models:/{name}/{version}')
            # Делаем предсказание
            logging.info(f"Делаем предобработку")
            self.pred = model.predict(test_df)
            self.res = {'predict': self.pred.tolist()}
            return {'predict': self.pred}
        except:
            logging.error("this model doesnt exist or invalid input data", exc_info=True)
            self.pred = 'this model doesnt exist or invalid input data'
            self.res = {'predict': self.pred}
            return {'predict': self.pred}

    def predict_model(self, input_data_predict, name):
        """
        Функция возвращает предсказание модели по данным, которые отправил пользователь.
        Будет использоваться в PUT запросе для получения предсказаний
        :param input_data_predict: данные, который отправил пользователь по API
        :param name: название модели, которая будет использоваться для предсказаний. Все доступные
        имена лежат в ./models
        :return: json с предсказаниями
        """
        logging.info(f"Стадия предсказания")
        logging.info(f"Имя модели с помощью которой хотим получить предсказание {name}")
        try:
            logging.info(f"Считываем данные для предсказания")
            data = pd.read_json(input_data_predict['data'])
            # Предобрабатываем входные данные
            logging.info(f"Делаем предобработку")
            test_df = self.preprocessing(data)
            logging.info(f"Импортируем модель model/{name}.pkl")
            # Импорт модели, которую будем использовать
            model = joblib.load(f'model/{name}.pkl')
            # Делаем предсказание
            logging.info(f"Делаем предобработку")
            self.pred = model.predict(test_df)
            self.res = {'predict': self.pred.tolist()}
            return {'predict': self.pred}
        except:
            logging.error("this model doesnt exist or invalid input data", exc_info=True)
            self.pred = 'this model doesnt exist or invalid input data'
            self.res = {'predict': self.pred}
            return {'predict': self.pred}

    def get_model_predict(self):
        """
        Функция будет использаваться в GET запросе от пользователя для получения предсказаний
        :return: Возвращает предсказание модели из predict_model
        """
        return self.res

    def update_model_name(self, update_data):
        """
        Функция обновляет название о модели в БД и в файле ./model
        :param update_data: dict - {old_name: ... , new_name: ...}
        :return: Возвращает сообщение со статусом выпоняемой задачи
        """
        logging.info(f"Меняем имя модели")
        db = sqlite3.connect('flask_app.db')
        sql = db.cursor()
        old_name = update_data['old_name']
        print(old_name)
        new_name = update_data['new_name']
        sql.execute(f'SELECT name FROM vvs_models_info WHERE name = "{old_name}.pkl"')

        if sql.fetchone() is not None:
            sql.execute(f'UPDATE vvs_models_info SET name = "{new_name}" WHERE name = "{old_name}.pkl"')
            db.commit()
            db.close()
            massege = 'success update'
            os.rename(f'model/{old_name}.pkl', f'model/{new_name}.pkl')

        else:
            massege = 'name doesnt exist'
        self.update_massege = {'massege': massege}
        return massege

    def update_model_info(self):
        """
        Вспомогательная функция для update_model_name
        """
        return self.update_massege

    def delete_model(self, name):
        """
        Функция используется для удалния модели из доступных моделей, которые лежат в ./model
        :param name: Имя модели, которую требуется удалить
        :return: message
        """
        logging.info(f"Пытаемся удалить модель {name}")
        try:
            os.remove(f'model/{name}.pkl')
            db = sqlite3.connect('flask_app.db')
            sql = db.cursor()
            sql.execute(f'DELETE FROM vvs_models_info WHERE "{name}.pkl"')
            db.commit()
            db.close()
            logging.info(f"Done")
            return 'successfully deleted'
        except:
            logging.error("file doesnt exist", exc_info=True)
            return 'file doesnt exist'


model = MLModels()
