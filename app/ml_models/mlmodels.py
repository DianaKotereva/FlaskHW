from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import numpy as np
import optuna
from sklearn.model_selection import cross_val_score


class Objective(object):

    """
    Класс для обучения одного запуска для оптимизации гиперпараметров с помощью optuna. 
    
    Принимает:
    
    train - Обучающие данные (pandas DataFrame)
    y_train - Обучающий таргет для бинарной классификации
    cv - количество фолдов для расчета метрик на кросс-валидации. Default: cv = 3
    optimize_metric - оптимизируемая метрика (одна из метрик из библиотеки sklearn). Default: optimize_metric = 'roc_auc'
    model_name - тип обучаемой модели: ['logreg', 'forest', 'boosting']. Default: model_name = 'logreg'
    config - пороги отбираемых гиперпараметров. Изначально в методе есть оптимизируемое по умолчанию пространство гиперпараметров, которое можно обновлять с помощью метода config. Default: config = {}.
    Параметр class_weight зафиксирован как 'balanced' и не подбирается. 
    
    Оптимизируемые параметры по умолчанию:
    
    Logreg: 'C', 'max_iter'
    RandomForestClassifier: 'max_depth', 'n_estimators', 'max_features', 'min_samples_leaf', 'min_samples_split','random_state'
    LGBMClassifier: 'max_depth', "num_leaves", 'learning_rate', 'feature_fraction', 'bagging_fraction', "bagging_freq", "lambda_l1", "lambda_l2", 'n_estimators'
    
    """
    
    def __init__(self, train, y_train, cv=3, optimize_metric='roc_auc', model_name='logreg', conf={}):

        self.available_models = ['logreg', 'forest', 'boosting']
        self.models_to_train = {'logreg': LogisticRegression,
                                'forest': RandomForestClassifier,
                                'boosting': LGBMClassifier}
        self.optimize_metric = optimize_metric

        self.model_name = model_name

        self.train = train
        self.y_train = y_train
        self.cv = cv

        self.default_params = {'logreg': {'class_weight': 'balanced', 'random_state': 241, 'penalty': 'l2',
                                          'solver': 'liblinear'},
                               'forest': {'class_weight': 'balanced', 'random_state': 241},
                               'boosting': {'class_weight': 'balanced', 'random_state': 241,
                                            'boosting_type': 'gbdt'}}

        if model_name == 'logreg':
            self.conf = {'C': [0.05, 4],
                         'class_weight': 'balanced',
                         'penalty': 'l2',
                         'random_state': 241,
                         'solver': 'liblinear',
                         'max_iter': [100, 500]}

        elif model_name == 'forest':
            self.conf = {'max_depth': [2, 5],
                         'class_weight': 'balanced',
                         'n_estimators': [50, 300],
                         'max_features': [0.15, 0.5],
                         'min_samples_leaf': [5, 10],
                         'min_samples_split': [10, 25],
                         'random_state': 241}

        elif model_name == 'boosting':
            self.conf = {'boosting_type': ['gbdt'],
                         'max_depth': [2, 3],
                         "num_leaves": [2, 150],
                         'learning_rate': [1e-5, 5e-1],
                         'feature_fraction': [0.4, 1.0],
                         'bagging_fraction': [0.4, 1.0],
                         "bagging_freq": [1, 7],
                         "lambda_l1": [1e-1, 10.0],
                         "lambda_l2": [1e-1, 10.0],
                         'verbose': 0,
                         'n_estimators': [30, 100],
                         'class_weight': 'balanced',
                         'random_state': 241
                         }
        self.conf.update(conf)

    def __call__(self, trial):

        config = {}
        for i in self.conf.keys():
            if type(self.conf[i]) is not list or len(self.conf[i]) == 1:
                config[i] = self.conf[i]
            elif len(self.conf[i]) > 2:
                config[i] = trial.suggest_categorical(i, self.conf[i])
            elif len(self.conf[i]) == 2:
                if type(self.conf[i][0]) is type(self.conf[i][1]) and type(self.conf[i][0]) is int:
                    config[i] = trial.suggest_int(i, self.conf[i][0], self.conf[i][1])
                elif type(self.conf[i][0]) in (float, int) and type(self.conf[i][1]) in (float, int):
                    config[i] = trial.suggest_loguniform(i, self.conf[i][0], self.conf[i][1])
                else:
                    raise KeyError(
                        f"Check data type {i}={self.conf[i]}, both values should be int, or float")
            else:
                raise KeyError(
                    f"Check data type for {i}={self.conf[i]}, should be edges or list of values")

        mean_cv_score = self.train_model(params=config)

        return mean_cv_score

    def train_model(self, params={}):

        params_to_fit = self.default_params[self.model_name]
        if 'class_weight' in list(params.keys()):
            params.pop('class_weight', None)
        params_to_fit.update(params)

        try:
            model = self.models_to_train[self.model_name](**params_to_fit)
        except KeyError:
            raise KeyError('Select model name from list = [logreg, forest, boosting]')
        except TypeError:
            raise KeyError('Unknown Parameters!')

        model.fit(self.train, self.y_train)
        res = np.mean(cross_val_score(model, self.train, self.y_train, cv=self.cv, scoring=self.optimize_metric))

        return res


class MLModelsDAO:
    
    """
    Класс для обучения моделей.
    =
    Параметры:
    
    ml_models - обновляемый список обученных моделей. На этапе инициализации ставится как пустой список
    available_models - доступные модели: ['logreg', 'forest', 'boosting']
    
    default_params - дефолтные гиперпараметры для моделей:
    {
    'logreg': {'class_weight': 'balanced', 'random_state': 241, 'penalty': 'l2', 'solver': 'liblinear'},
    'forest': {'class_weight': 'balanced', 'random_state': 241},
    'boosting': {'class_weight': 'balanced', 'random_state': 241, 'boosting_type': 'gbdt'}}
    
    cv - количество фолдов для оценки качества модели на cross_val_score. При cv=1 качество оценивается только train данным, без биения на фолды (рекомендуется использовать cv=1 только для ОЧЕНЬ маленьких данных).
    
    Методы:
    
    predict - возвращает прогноз для тестовых данных
    train - обучение новой модели
    retrain - переобучение существующей модели
    delete - удалить модель из списка
    get - получить модель из списка
    find_hyperparams - подбор гиперпараметров с помощью optuna
    
    """
    
    def __init__(self):
        self.ml_models = []
        self.counter = len(self.ml_models)
        self.available_models = ['logreg', 'forest', 'boosting']
        self.models_to_train = {'logreg': LogisticRegression,
                                'forest': RandomForestClassifier,
                                'boosting': LGBMClassifier}
        self.default_params = {'logreg': {'class_weight': 'balanced', 'random_state': 241, 'penalty': 'l2',
                                          'solver': 'liblinear'},
                               'forest': {'class_weight': 'balanced', 'random_state': 241},
                               'boosting': {'class_weight': 'balanced', 'random_state': 241,
                                            'boosting_type': 'gbdt'}}
        self.cv = 3

    def predict(self, ids, test, y_test=None):
        
        """
        Получить прогноз модели
        
        Параметры: 
        
        ids - индекс модели, для которой требуется получить прогноз. ids не должен превышать общее количество обученных моделей!
        test - тестовые данные
        y_test - таргет для тестовых данных. При использовании y_test класс оценивает качество прогноза для тестовых данных и возвращает его. При отсутствии y_test - класс возвращает вектор predict_proba
        
        """

        models = self.get(ids)
        predictions = models['model'].predict_proba(test)[:, 1]
        predicts = models['model'].predict(test)

        if (y_test is not None) and (type(y_test) != int) and (
                type(y_test) != int and hasattr(y_test, '__len__') and len(y_test) > 0):
            
            if len(y_test) != len(predicts):
                raise KeyError('Data and Target shapes are not equal!')
            else:
                accuracy = accuracy_score(y_test, predicts)
                roc_auc = roc_auc_score(y_test, predictions)
                aps = average_precision_score(y_test, predictions)
                res = {'predict_proba': list(predictions), 'acc': accuracy, 'roc_auc': roc_auc, 'aps': aps}
        else:
            res = list(predictions)
        return res

    def train_model(self, model_name,
                    train, y_train, test=None, y_test=None,
                    params={}, cv=None):
        
        """
        Метод для обучения модели
        
        Параметры:
        
        train - обучающие данные
        y_train - таргет
        test - тестовые данные (опционально). При их использовании класс расчитывает метрики на тестовых данных. При их отсутствии - метрики расчитываются на кросс-валидации
        y_test - таргет для тестовых данных (опционально). При их использовании класс расчитывает метрики на тестовых данных. При их отсутствии - метрики расчитываются на кросс-валидации
        params - гиперпараметры. Default: {}
        cv - количество фолдов для оценки качества модели на cross_val_score. При cv=1 качество оценивается только train данным, без биения на фолды (рекомендуется использовать cv=1 только для ОЧЕНЬ маленьких данных).
        
        Возвращает:
        
        res = {'model': model,
               'name': model_name, 'params': params_to_fit,
               'acc': accuracy, 'roc_auc': roc_auc, 'aps': aps}
        
        """

        if cv is None:
            cv = self.cv
        
        params_to_fit = self.default_params[model_name]
        params_to_fit.update(params)
        
        if train.shape[0] != y_train.shape[0]:
            raise KeyError('Data and Target shapes are not equal!')
            
        try:
            model = self.models_to_train[model_name](**params_to_fit)
        except KeyError:
            raise KeyError('Select model name from list = [logreg, forest, boosting]')
        except TypeError:
            raise KeyError('Unknown Parameters!')

        model.fit(train, y_train)

        if test is not None and (y_test is not None and (y_test is not None)
                                 and (type(y_test) != int) and
                                 (type(y_test) != int and hasattr(y_test, '__len__') and len(y_test) > 0)):

            predictions = model.predict_proba(test)[:, 1]
            predicts = model.predict(test)

            accuracy = accuracy_score(y_test, predicts)
            roc_auc = roc_auc_score(y_test, predictions)
            aps = average_precision_score(y_test, predictions)

        else:
            if cv < 2:
                predictions = model.predict_proba(train)[:, 1]
                predicts = model.predict(train)

                accuracy = accuracy_score(y_train, predicts)
                roc_auc = roc_auc_score(y_train, predictions)
                aps = average_precision_score(y_train, predictions)
            else:
                accuracy = np.mean(cross_val_score(model, train, y_train, cv=cv, scoring='accuracy'))
                roc_auc = np.mean(cross_val_score(model, train, y_train, cv=cv, scoring='roc_auc'))
                aps = np.mean(cross_val_score(model, train, y_train, cv=cv, scoring='average_precision'))

        res = {'model': model,
               'name': model_name, 'params': params_to_fit,
               'acc': accuracy, 'roc_auc': roc_auc, 'aps': aps}

        return res

    def add_to_list(self, res, id_num=None):
        
        """
        Добавить обученную модель в список моделей.
        
        """
        if id_num is None:
            self.ml_models.append(res)
        else:
            if id_num < len(self.ml_models):
                self.ml_models[id_num] = res
            else:
                raise IndexError(f'Only {len(self.ml_models)} models are trained!')
        self.counter = len(self.ml_models)

    def train(self, model_name,
              train, y_train, test=None, y_test=None,
              params={}, ids=None, cv=None):
        
        """ 
        Обучить модель и добавить ее в список обученных моделей.
        
        Параметры:
        
        train - обучающие данные
        y_train - таргет
        test - тестовые данные (опционально). При их использовании класс расчитывает метрики на тестовых данных. При их отсутствии - метрики расчитываются на кросс-валидации
        y_test - таргет для тестовых данных (опционально). При их использовании класс расчитывает метрики на тестовых данных. При их отсутствии - метрики расчитываются на кросс-валидации
        params - гиперпараметры. Default: {}
        cv - количество фолдов для оценки качества модели на cross_val_score. При cv=1 качество оценивается только train данным, без биения на фолды (рекомендуется использовать cv=1 только для ОЧЕНЬ маленьких данных).
        
        Возвращает:
        
        res = {'model': model,
               'name': model_name, 'params': params_to_fit,
               'acc': accuracy, 'roc_auc': roc_auc, 'aps': aps}
        
        """
        
        res = self.train_model(model_name,
                               train, y_train, test, y_test,
                               params=params, cv=cv)
        self.add_to_list(res, ids)
        return {'acc': res['acc'], 'roc_auc': res['roc_auc'], 'aps': res['aps']}

    def retrain(self, ids, train, y_train, test, y_test, params=None, cv = None):
        
        """
        Переобучить модель
        
        Параметры:
        
        ids - номер изменяемой модели
        train - обучающие данные
        y_train - таргет
        params - гиперпараметры. Default: {}
        cv - количество фолдов для оценки качества модели на cross_val_score. При cv=1 качество оценивается только train данным, без биения на фолды (рекомендуется использовать cv=1 только для ОЧЕНЬ маленьких данных).
        
        Возвращает:
        
        res = {'model': model,
               'name': model_name, 'params': params_to_fit,
               'acc': accuracy, 'roc_auc': roc_auc, 'aps': aps}
        
        """
        
        res = self.get(ids)
        if params is None:
            params = res['params']
        r = self.train(model_name=res['name'],
                       train=train, y_train=y_train, test=test, y_test=y_test,
                       params=params,
                       ids=ids, cv=cv)
        return r

    def delete(self, ids):
        
        """
        Удалить модель из списка
        
        Параметры: 
        
        ids - номер модели на удаление
        
        """
        
        if ids < len(self.ml_models):
            self.ml_models.pop(ids)
            self.counter = len(self.ml_models)
        else:
            raise IndexError(f'Only {len(self.ml_models)} models are trained!')

    def get(self, ids):
        
        """
        Получить модель
        
        Параметры:
        
        ids - номер модели
        
        """
        
        if ids < len(self.ml_models):
            return self.ml_models[ids]
        raise IndexError(f'Only {len(self.ml_models)} models are trained!')

    def find_hyperparams(self, model_name, train, y_train, cv=3, optimize_metric='roc_auc',
                         conf={}, n_trials=20, optimizer_direction='maximize'):

        """
        Функция для подбора гиперпараметров с помощью метода optuna
        
        Параметры:
        
        train - обучающие данные
        y_train - таргет
        cv - количество фолдов для оценки качества модели на cross_val_score. Default: 3.При cv=1 качество оценивается только train данным, без биения на фолды (рекомендуется использовать cv=1 только для ОЧЕНЬ маленьких данных).
        optimize_metric - оптимизируемая метрика (одна из метрик из библиотеки sklearn). Default: optimize_metric = 'roc_auc'
        conf - границы пространства гиперпараметров. Default: {}
        n_trials - количество итераций оптимизации. Default: 20 
        optimizer_direction - направление оптимизации ('maximize'/'minimize'). Default: 'maximize'
        
        """
        
        fixed_params = {}
        for k, val in conf.items():
            if type(val) != list:
                fixed_params[k] = val
            elif type(val) == list and len(val) == 1:
                fixed_params[k] = val[0]

        objective = Objective(train=train, y_train=y_train,
                              cv=cv, optimize_metric=optimize_metric,
                              model_name=model_name, conf=conf)
        study = optuna.create_study(direction=optimizer_direction)
        study.optimize(objective, n_trials=n_trials)

        best_params = study.best_params
        best_params.update(fixed_params)

        return best_params
