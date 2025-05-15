
import streamlit as st
from deeppavlov import build_model, configs
import telegram
from aiogram import Bot, Dispatcher, types, executor

# Инициализация модели Deep Pavlov
model = build_model(configs.faq.tfidf_logreg_en_faq, download=True)

st.title('Чат-бот на основе ML')

user_question = st.text_input('Введите ваш вопрос:')

if user_question:
    response = model([user_question])[0]
    st.write(f'Ответ: {response}')

# Телеграм бот
TELEGRAM_API_TOKEN = st.secrets['TELEGRAM_API_TOKEN']
bot = Bot(token=TELEGRAM_API_TOKEN)
dp = Dispatcher(bot)

@dp.message_handler(commands=['start'])
async def send_welcome(message: types.Message):
    await message.reply("Привет! Я ML чат-бот. Задайте мне вопрос.")

@dp.message_handler()
async def answer_question(message: types.Message):
    response = model([message.text])[0]
    await message.reply(response)

if __name__ == '__main__':
    executor.start_polling(dp, skip_updates=True)
    