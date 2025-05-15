import streamlit as st
from deeppavlov import build_model, configs
from aiogram import Bot, Dispatcher, types, executor
import os

# Проверяем наличие модели, если её нет, скачиваем её
try:
    model = build_model(configs.faq.tfidf_logreg_en_faq, download=True)
except AttributeError:
    st.error("Ошибка: Модель tfidf_logreg_en_faq недоступна. Проверьте версию DeepPavlov.")

st.title('Чат-бот на основе ML')

user_question = st.text_input('Введите ваш вопрос:')

if user_question:
    response = model([user_question])[0]
    st.write(f'Ответ: {response}')

# Телеграм бот
TELEGRAM_API_TOKEN = os.getenv("TELEGRAM_API_TOKEN")

if TELEGRAM_API_TOKEN:
    bot = Bot(token=TELEGRAM_API_TOKEN)
    dp = Dispatcher(bot)

    @dp.message_handler(commands=['start'])
    async def send_welcome(message: types.Message):
        await message.reply("Привет! Я ML чат-бот. Задайте мне вопрос.")

    @dp.message_handler()
    async def answer_question(message: types.Message):
        response = model([message.text])[0]
        await message.reply(response)

    st.write("Телеграм бот настроен и готов к работе.")
else:
    st.warning("Telegram бот не настроен. Убедитесь, что TELEGRAM_API_TOKEN установлен в переменных окружения.")

if __name__ == '__main__':
    st.write("Запуск приложения...")
