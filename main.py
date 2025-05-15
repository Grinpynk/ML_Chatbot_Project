"""
Чат-бот на основе ML (Hugging Face Transformers)
Разработан для ответов на вопросы на основе заданного текста.
"""

import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
from requests.exceptions import HTTPError

st.title('Чат-бот на основе ML (Hugging Face Transformers)')

@st.cache_resource
def load_model():
    """
    Загружает модель и токенизатор для выполнения задачи вопрос-ответ (Q&A).
    Возвращает pipeline модели Hugging Face.
    """
    model_name = "distilbert-base-uncased-distilled-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_model

qa_pipeline = load_model()

user_question = st.text_input('Введите ваш вопрос:')
context_text = st.text_area('Введите текст или контекст, на основе которого будет дан ответ:')

if user_question and context_text:
    try:
        result = qa_pipeline(question=user_question, context=context_text)
        st.write(f'Ответ: {result["answer"]}')
    except ValueError:
        st.error('Ошибка: некорректный ввод данных. Проверьте вопрос и контекст.')
    except RuntimeError:
        st.error('Ошибка: не удалось загрузить модель. Попробуйте позже.')
    except KeyError:
        st.error('Ошибка: проблема с обработкой ответа. Попробуйте изменить вопрос.')
    except HTTPError:
        st.error('Ошибка сети: проверьте подключение к Интернету.')
else:
    st.info('Пожалуйста, введите вопрос и контекст для получения ответа.')
