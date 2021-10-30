import telebot

from requests import get

bot = telebot.TeleBot('2077078112:AAF2keca4msLO-0IFeaRIlMhoz7yfUmC090')

@bot.message_handler(content_types=['text'])
def get_text_messages(message):
    if message.text == "Привет":
        bot.send_message(message.from_user.id, "Привет, чем я могу тебе помочь?")
    elif message.text == "/help":
        bot.send_message(message.from_user.id, "Напиши привет")
    else:
        bot.send_message(message.from_user.id, "Я тебя не понимаю. Напиши /help.")
        
bot.polling(none_stop=True, interval=0)
