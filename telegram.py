import telebot
from telebot import types
import requests

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score
import numpy as np
import pandas as pd
import optuna
from sklearn.model_selection import StratifiedKFold, cross_val_score
import json
import pandas as pd

bot = telebot.TeleBot('XXXXXXXXXXXXXXXXXXXXXXXXXXX')

model_classes = {'logreg': 'Логистическая регрессия', 'forest': 'RandomForestClassifier', 'boosting': 'LGBMClassifier'}
model_classes_inverse = {v:k for k, v in model_classes.items()}

train = None
y_train = None
test = None
y_test = None
step = ''
config = str({})
find_params = None
cv = 3
n_trials = 20
pass_model = 'logreg'
number = 0
to_do = ''

# global train
# global y_train
# global test
# global y_test

# global config
# global find_params
# global cv
# global n_trials
# global pass_model

# global step
# global number
# global to_do

num = 0

@bot.message_handler(content_types=['text', 'document'])
def start(message):
    bot.send_message(message.from_user.id, 
                         "Привет, я ML бот! Я умею обучать ML модели. Введите /start, поучим модели!")
    if message.text == '/start':
        bot.register_next_step_handler(message, buttons) 
    else:
        bot.send_message(message.from_user.id, 'Напиши /start');
        
def buttons(message):
    
    global num
    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    
    bt1 = types.KeyboardButton('Какие модели умеешь обучать?')
    bt2 = types.KeyboardButton('Обучить модель')
    bt3 = types.KeyboardButton('Какие модели обучены?')
    bt4 = types.KeyboardButton('Сколько моделей обучено?')
    bt5 = types.KeyboardButton('Получить модель')
    bt6 = types.KeyboardButton('Прогноз')
    bt7 = types.KeyboardButton('Delete')
    
    markup.add(bt1, bt2, bt3, bt4, bt5, bt6, bt7)
    
    if num == 0:
        question = 'Выберите, что вы хотите?'
    else:
        question = 'Что-то еще?'
    
    msg = bot.send_message(message.chat.id, text=question, reply_markup=markup)
    num+=1
    bot.register_next_step_handler(msg, make_res)
    
def make_res(message):
    
    global train
    global y_train
    global test
    global y_test
    
    global config
    global find_params
    global cv
    global n_trials
    global pass_model
    
    global step
    global number
    global to_do
    
    step = ''
    find_params = None
    config = str({})
    to_do = ''
    
    if message.text == 'Какие модели обучены?':
        get_models(message)
    elif message.text == 'Сколько моделей обучено?':
        get_count_models(message)
    elif message.text == 'Какие модели умеешь обучать?':
        bot.send_message(message.chat.id, 
                         "Я умею обучать модели бинарной классификации: Логистическую регрессию, Случайный лес и Градиентный Бустинг")
        buttons(message)
    elif message.text == 'Получить модель':
        to_do = 'get'
        bot.send_message(message.chat.id, 'Нужен номер модели')
        select_one(message)
    elif message.text == 'Delete':
        to_do = 'delete'
        delete_choice(message)
    elif message.text == 'Обучить модель':
        to_do = 'train'
        train_choice(message)
    elif message.text == 'Прогноз':
        to_do = 'predict'
        bot.send_message(message.chat.id, 'Нужен номер модели')
        select_one(message)
        
def train_choice(message):
    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    bt1 = types.KeyboardButton('Новая модель')
    bt2 = types.KeyboardButton('Переобучить модель')
    markup.add(bt1, bt2)
    
    question = 'Что вы хотите обучить?'
    msg = bot.send_message(message.chat.id, text=question, reply_markup=markup)
    bot.register_next_step_handler(msg, train_sel)
    
def train_sel(message):
    global to_do
    
    if message.text == 'Новая модель':
        to_do = 'train_new'
        markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
        bt1 = types.KeyboardButton('Логистическая регрессия')
        bt2 = types.KeyboardButton('RandomForestClassifier')
        bt3 = types.KeyboardButton('LGBMClassifier')
        markup.add(bt1, bt2, bt3)

        question = 'Что вы хотите обучить?'
        msg = bot.send_message(message.chat.id, text=question, reply_markup=markup)
        bot.register_next_step_handler(message, select_type)
        
    elif message.text == 'Переобучить модель':
        to_do = 'retrain'
        bot.send_message(message.chat.id, 'Какую модель? Нужен номер изменяемой модели')
        select_one(message)
        
def select_type(message):
    
    global step
    global pass_model
    
    step = 'Обучить модель'
    if to_do == 'train_new':
        pass_model = model_classes_inverse[message.text]
    
    bot.send_message(message.chat.id, 'Проверим train данные')
    if train is None or y_train is None:
        bot.send_message(message.chat.id, 'Train данных нет. Пожалуйста, пришлите train датасет (без таргета) в формате json:')
        bot.register_next_step_handler(message, get_train_data)
    else:
        yes_no_choice(message)
            
def yes_no_choice(message):  
    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    bt1 = types.KeyboardButton('Да')
    bt2 = types.KeyboardButton('Нет')
    markup.add(bt1, bt2)
    
    if step == 'Обучить модель':
        question = 'Сейчас в памяти имеются train данные. Хотите поменять?'
    elif step == 'Подобрать гиперпараметры':
        question = 'Вы хотите подобрать гиперпараметры с помощью optuna?'
    elif step == 'Дополнительные параметры подбора':
        question = 'Вы хотите задать cv и n_trials (для optuna). Default: cv = 3, n_trials = 20? Если датасет маленький, лучше поставить cv=1'
    elif step == 'Задать гиперпараметры':
        question = 'Вы хотите задать свои гиперпараметры для подбора/обучения?'
    elif step == 'Получить тестовые данные':
        question = 'Вы хотите использовать y_test для оценки?'
    
    msg = bot.send_message(message.chat.id, text=question, reply_markup=markup)
    bot.register_next_step_handler(msg, check_yes_no)
    
def check_yes_no(message):
    
    global step
    global find_params
    global y_test
    
    if step == 'Обучить модель':
        if message.text == 'Да':
            bot.send_message(message.chat.id, 'Хорошо. Пожалуйста, пришлите train датасет (без таргета) в формате json')
            bot.register_next_step_handler(message, get_train_data)
        elif message.text == 'Нет':
            step = 'Подобрать гиперпараметры'
            yes_no_choice(message)
            
    elif step == 'Подобрать гиперпараметры':
        if message.text == 'Да':
            find_params = True
            step = 'Дополнительные параметры подбора'
            yes_no_choice(message)
        elif message.text == 'Нет':
            find_params = None
            step = 'Дополнительные параметры подбора'
            yes_no_choice(message)
    
    elif step == 'Дополнительные параметры подбора':
        if message.text == 'Да':
            bot.send_message(message.chat.id, 'Хорошо. Введите cv')
            bot.register_next_step_handler(message, get_cv)
        elif message.text == 'Нет':
            step = 'Задать гиперпараметры'
            yes_no_choice(message)
            
    elif step == 'Задать гиперпараметры':
        if message.text == 'Да':
            bot.send_message(message.chat.id, 'Хорошо. Пожалуйста, введите строку с config')
            bot.register_next_step_handler(message, get_config)
        else:
            bot.send_message(message.chat.id, 'Хорошо. Начинаю обучать модель')
            train_model(message)
    
    elif step == 'Получить тестовые данные':
        if message.text == 'Да':
            bot.send_message(message.chat.id, 'Хорошо. Пожалуйста, пришлите y_test array в формате json')
            bot.register_next_step_handler(message, get_y_test_data)      
        else:
            y_test = None
            bot.send_message(message.chat.id, 'Хорошо. Начинаю прогноз')
            predict(message)
            
    
def get_cv(message):
    global cv
    try:
        cv = int(message.text)
        bot.send_message(message.chat.id, 'Хорошо. Введите n_trials')
        bot.register_next_step_handler(message, get_n_trials)
    except ValueError:
        bot.send_message(message.chat.id, 'Цифрами, пожалуйста')
        bot.register_next_step_handler(message, get_cv)
            
def get_n_trials(message):
    
    global step
    global n_trials
    try:
        n_trials = int(message.text)
        step = 'Задать гиперпараметры'
        yes_no_choice(message)
    except ValueError:
        bot.send_message(message.chat.id, 'Цифрами, пожалуйста')
        bot.register_next_step_handler(message, get_n_trials)            
    
def get_config(message):
    config = message.text
    bot.send_message(message.chat.id, 'Хорошо. Начинаю обучать модель')
    train_model(message)
    
def get_train_data(message):
    
    global train
    try:
        if message.content_type == 'document':
            chat_id = message.chat.id
            file_info = bot.get_file(message.document.file_id)
            train = bot.download_file(file_info.file_path)
            train = train.decode('utf-8')
        
        elif message.content_type == 'text':
            train = message.text
        
        train = pd.DataFrame(json.loads(train))
        
        bot.send_message(message.chat.id, 'Хорошо. Пожалуйста, пришлите y_train array в формате json')
        bot.register_next_step_handler(message, get_y_train_data)         
        
    except Exception as e:
        bot.send_message(message.chat.id, 'Не понимаю, что вы прислали. Пожалуйста, пришлите train данные в формате json')
        bot.register_next_step_handler(message, get_train_data)          
    
def get_y_train_data(message):
    global step
    global y_train
    try:
        if message.content_type == 'document':
            chat_id = message.chat.id
            file_info = bot.get_file(message.document.file_id)
            y_train = bot.download_file(file_info.file_path)
            y_train = y_train.decode('utf-8')

        elif message.content_type == 'text':
            y_train = message.text

        y_train = eval(y_train)
        if (len(y_train) == 1 and type(y_train[list(y_train.keys())[0]]) == dict):
            y_train = y_train[list(y_train.keys())[0]]
        y_train = pd.Series(y_train)

        if len(np.unique(y_train)) > 2:
            bot.send_message(message.chat.id, 'Класс обучает БИНАРНЫЕ классификации! Пришлите таргет еще раз')
            bot.register_next_step_handler(message, get_y_train_data)
        else:
            step = 'Подобрать гиперпараметры'
            yes_no_choice(message)

    except Exception as e:
        bot.send_message(message.chat.id, 'Не понимаю, что вы прислали. Пожалуйста, пришлите y_train array в формате json')
        bot.register_next_step_handler(message, get_y_train_data)      
        
def get_test_data(message):
    
    global step
    global test
    try:
        if message.content_type == 'document':
            chat_id = message.chat.id
            file_info = bot.get_file(message.document.file_id)
            test = bot.download_file(file_info.file_path)
            test = test.decode('utf-8')
        elif message.content_type == 'text':
            test = message.text
        
        test = pd.DataFrame(json.loads(test))
        step = 'Получить тестовые данные'
        yes_no_choice(message)
    
    except Exception as e:
        bot.send_message(message.chat.id, 'Не понимаю, что вы прислали. Пожалуйста, пришлите test данные в формате json')
        bot.register_next_step_handler(message, get_test_data)  
    
    
def get_y_test_data(message):
    global step
    global y_test
    try:
        if message.content_type == 'document':
            chat_id = message.chat.id
            file_info = bot.get_file(message.document.file_id)
            y_test = bot.download_file(file_info.file_path)
            y_test = y_test.decode('utf-8')
        
        elif message.content_type == 'text':
            y_test = message.text
        
        y_test = eval(y_test)
        if (len(y_test) == 1 and type(y_test[list(y_test.keys())[0]]) == dict):
            y_test = y_test[list(y_test.keys())[0]]
        y_test = pd.Series(y_test)
        
        if len(np.unique(y_test)) > 2:
            bot.send_message(message.chat.id, 'Класс обучает БИНАРНЫЕ классификации! Пришлите таргет еще раз')
            bot.register_next_step_handler(message, get_y_test_data)
        else:
            bot.send_message(message.chat.id, 'Начинаю прогноз')
            predict(message)
    except Exception as e:
        bot.send_message(message.chat.id, 'Не понимаю, что вы прислали. Пожалуйста, пришлите y_test array в формате json')
        bot.register_next_step_handler(message, get_y_test_data)   
        
    
def predict(message):
    
    global number
    try:
        url = "http://localhost:5000/api/ml_models/predict/"+str(number)
        if y_test is not None:
            resp = requests.get(url, data = {'test': test.to_json(), 'y_test': y_test.to_json()})
        else:
            resp = requests.get(url, data = {'test': test.to_json()})
        res = eval(resp.text)
        if type(res) == list:
            ## Отправить в виде json/txt файла
            with open('predict.txt', 'w') as file:
                file.write(str(res))
            f = open("predict.txt","rb")
            bot.send_document(message.chat.id,f)

        elif type(res) == dict:
            ## Отправить predict_proba в виде json/txt файла
            ## Напечатать результат
            if 'message' in list(res.keys()):
                bot.send_message(message.chat.id, f"Ошибка: {res['message']}")
            else:
                with open('predict.txt', 'w') as file:
                    file.write(str(res['predict_proba']))
                f = open("predict.txt","rb")
                bot.send_document(message.chat.id,f)
                bot.send_message(message.chat.id, f"Модель {number}. \n Accuracy: {res['acc']}. \n ROC_AUC: {res['roc_auc']}. \n APS: {res['aps']}")

        buttons(message)
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")
        buttons(message)
    
def train_model(message):
    
    print(find_params)
    print(config)
    print(n_trials)
    print(to_do)
    
    global train
    global y_train
    
    try:
        if to_do == 'train_new':
            resp = requests.post("http://localhost:5000/api/ml_models/train", data = {'train': train.to_json(), 
                                                                                 'y_train': y_train.to_json(),
                                                                                 'name': pass_model,
                                                                                 'find_params': find_params,
                                                                                 'n_trials': n_trials,
                                                                                 'cv':cv, 
                                                                                 'params': config, 
                                                                                 'config': config})
            res = eval(resp.text)
        elif to_do == 'retrain':
            url = "http://localhost:5000/api/ml_models/"+str(number)
            resp = requests.put(url, data = {'train': train.to_json(), 
                                             'y_train': y_train.to_json(),
                                             'name': pass_model,
                                             'find_params': find_params,
                                             'n_trials': n_trials,
                                             'cv':cv, 
                                             'params': config, 
                                             'config': config})
            res = eval(resp.text)

        if 'message' in list(res.keys()):
            bot.send_message(message.chat.id, f"Модель не обучена, ошибка: {res['message']}")

            train = None
            y_train = None

        else:
            bot.send_message(message.chat.id, f"Модель обучена\n Accuracy: {res['acc']}. \n ROC_AUC: {res['roc_auc']}. \n APS: {res['aps']}")
        buttons(message)
    except Exception as e:
        bot.send_message(message.chat.id, f"Error: {e}")
        buttons(message)
    
        
def get_models(message):
    resp = requests.get("http://localhost:5000/api/ml_models")
    res = eval(eval(resp.text))
    if len(res) > 0:
        for num in range(len(res)):
            r = res[num]
            bot.send_message(message.chat.id, 
                             f"Модель {num}. Тип: {model_classes[r['name']]}. \n Accuracy: {r['acc']}. \n ROC_AUC: {r['roc_auc']}. \n APS: {r['aps']}")
    else:
        bot.send_message(message.chat.id, 
                             "Еще не обучено ни одной модели. Самое время начать!")
    buttons(message)
    
def get_count_models(message):
    resp = requests.get("http://localhost:5000/api/ml_models/count")
    res = eval(resp.text)
    bot.send_message(message.chat.id, res)
    buttons(message)
    
def delete_choice(message):
    
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    bt1 = types.KeyboardButton('Все')
    bt2 = types.KeyboardButton('Одну модель')
    markup.add(bt1, bt2)
    
    question = 'Что вы хотите удалить?'
    msg = bot.send_message(message.chat.id, text=question, reply_markup=markup)
    bot.register_next_step_handler(msg, delete)

def delete(message):
    if message.text == 'Все':
        resp = requests.delete("http://localhost:5000/api/ml_models")
        bot.send_message(message.chat.id, 'Все модели удалены, возвращаюсь на стартовое меню')
        buttons(message)
        
    elif message.text == 'Одну модель':
        bot.send_message(message.chat.id, 'Какую модель? Нужен номер удаляемой модели')
        select_one(message)

def select_one(message):
    markup = types.ReplyKeyboardMarkup(one_time_keyboard=True, resize_keyboard=True)
    resp = requests.get("http://localhost:5000/api/ml_models/count")
    res = eval(resp.text)
    if res == 0:
        bot.send_message(message.chat.id, 'Простите, но пока не обучено ни одной модели.')
        buttons(message)
    else:
        btts = []
        for n in range(res):
            bt1 = types.KeyboardButton(str(n))
            btts.append(bt1)

        markup.add(*btts)

        question = 'Выберите модель'
        msg = bot.send_message(message.chat.id, text=question, reply_markup=markup)
        bot.register_next_step_handler(message, process_one)

def process_one(message):
    global number
    
    print(to_do)
    number = int(message.text)
    url = "http://localhost:5000/api/ml_models/"+str(number)
    if to_do == 'delete':
        resp = requests.delete(url)
        bot.send_message(message.chat.id, f'Модель {number} удалена')
        buttons(message)

    elif to_do == 'retrain':
        bot.register_next_step_handler(message, select_type)

    elif to_do == 'get':
        resp = requests.get(url)
        r = eval(eval(resp.text))
        bot.send_message(message.chat.id, f"Модель {number}. Тип: {model_classes[r['name']]}. \n Accuracy: {r['acc']}. \n ROC_AUC: {r['roc_auc']}. \n APS: {r['aps']}")
        buttons(message)
        
    elif to_do == 'predict':
        bot.send_message(message.chat.id, 'Загрузите test данные в формате json')
        bot.register_next_step_handler(message, get_test_data)  
        
bot.polling(none_stop=True, interval=0)
