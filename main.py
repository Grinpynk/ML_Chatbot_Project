
import streamlit as st
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

st.title('Чат-бот на основе ML (Hugging Face Transformers)')

# Загружаем модель и токенизатор
@st.cache_resource
def load_model():
    model_name = "distilbert-base-uncased-distilled-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_pipeline

qa_pipeline = load_model()

user_question = st.text_input('Введите ваш вопрос:')
context_text = st.text_area('Введите текст или контекст, на основе которого будет дан ответ:')

if user_question and context_text:
    result = qa_pipeline(question=user_question, context=context_text)
    st.write(f'Ответ: {result["answer"]}')
