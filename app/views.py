from flask import request
from flask_restx import Resource
from app import api, models_dao
import logging
import pandas as pd

log = logging.getLogger(__name__)


@api.route('/api/ml_models')
class MLModels(Resource):

    def get(self):
        log.info('Get all models')
        return str(models_dao.ml_models)

    def delete(self):
        log.info('Clear all models')
        models_dao.ml_models = []


@api.route('/api/ml_models/can_train')
class MLModels(Resource):

    def get(self):
        log.info('Get list of all models API can train')
        return models_dao.available_models


@api.route('/api/ml_models/count')
class MLModels(Resource):

    def get(self):
        log.info('Get all trained models count')
        return len(models_dao.ml_models)


@api.route('/api/ml_models/<int:id>')
class MLModel(Resource):

    def get(self, id):
        log.info('Get model')
        log.info(f'id = {id}\n type(id) = {type(id)}')
        try:
            return str(models_dao.get(id))
        except NotImplementedError as e:
            api.abort(404, e)
        except KeyError as e:
            api.abort(404, e)
        except Exception as e:
            api.abort(404, e)

    def put(self, id):
        log.info('Retrain model')
        log.info(f'id = {id}\n type(id) = {type(id)}')

        try:

            train = pd.DataFrame(eval(request.form.get('train')))
            y_train = pd.Series(eval(request.form.get('y_train')))
            test = request.form.get('test')
            y_test = request.form.get('y_test')
            name = request.form.get('name')
            params = request.form.get('params')
            cv = request.form.get('cv')
            if cv is None:
                cv = 3
            else:
                cv = eval(cv)
            
            if params is None:
                params = {}
            else:
                params = eval(params)

            assert train is not None
            assert y_train is not None

            find_params = request.form.get('find_params')
            if find_params is not None:
                config = request.form.get('config')
                if config is None:
                    config = {}
                else:
                    config = eval(config)

                n_trials = request.form.get('n_trials')
                if n_trials is None:
                    n_trials = 30
                else:
                    n_trials = eval(n_trials)

            assert train is not None
            assert y_train is not None

            if test is not None:
                test = pd.DataFrame(eval(test))
                y_test = pd.Series(eval(y_test))

            if find_params:
                log.info('Find hyperparameters')
                best_params = models_dao.find_hyperparams(name, train, y_train, n_trials=n_trials, cv=cv, conf=config)
                params = best_params
            r = models_dao.retrain(id, train, y_train, test, y_test, params=params, cv=cv)
            return r

        except IndexError as e:
            api.abort(404, e)
        except Exception as e:
            api.abort(404, e)
    def delete(self, id):
        log.info('Delete model')
        log.info(f'id = {id}\n type(id) = {type(id)}')

        try:
            models_dao.delete(id)
        except IndexError as e:
            api.abort(404, e)
        except Exception as e:
            api.abort(404, e)

@api.route('/api/ml_models/train')
class MLModel(Resource):

    def post(self):
        log.info('Train new model')
        try:
            train = pd.DataFrame(eval(request.form.get('train')))
            y_train = pd.Series(eval(request.form.get('y_train')))
            test = request.form.get('test')
            y_test = request.form.get('y_test')
            name = request.form.get('name')
            params = request.form.get('params')
            cv = request.form.get('cv')
            if cv is None:
                cv = 3
            else:
                cv = eval(cv)
            
            if params is None:
                params = {}
            else:
                params = eval(params)

            find_params = request.form.get('find_params')
            if find_params is not None:
                config = request.form.get('config')
                if config is None:
                    config = {}
                else:
                    config = eval(config)

                n_trials = request.form.get('n_trials')
                if n_trials is None:
                    n_trials = 30
                else:
                    n_trials = eval(n_trials)

            assert train is not None
            assert y_train is not None

            if test is not None:
                test = pd.DataFrame(eval(test))
                y_test = pd.Series(eval(y_test))

            if find_params:
                log.info('Find hyperparameters')
                best_params = models_dao.find_hyperparams(name, train, y_train, n_trials=n_trials, cv=cv, conf=config)
                params = best_params
            r = models_dao.train(name, train, y_train, test, y_test, params=params, cv=cv)
            return r
        except KeyError as e:
            api.abort(404, e)
        except Exception as e:
            api.abort(404, e)


@api.route('/api/ml_models/predict/<int:id>')
class MLModel(Resource):

    def get(self, id):
        log.info('Predict')
        log.info(f'id = {id}\n type(id) = {type(id)}')

        try:
            test = pd.DataFrame(eval(request.form.get('test')))
        except ValueError:
            test = pd.DataFrame(pd.Series(eval(request.form.get('test'))))

        if test.shape[1] == 1:
            test = test.T
        y_test = request.form.get('y_test')
        if y_test is not None and type(y_test) != int:
            y_test = pd.Series(eval(y_test))
        try:
            return models_dao.predict(id, test, y_test)
        except IndexError as e:
            api.abort(404, e)
        except KeyError as e:
            api.abort(404, e)
        except Exception as e:
            api.abort(404, e)
