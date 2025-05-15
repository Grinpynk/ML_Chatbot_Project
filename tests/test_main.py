import pytest
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def test_qa_pipeline():
    # Загружаем модель для тестирования
    model_name = "distilbert-base-uncased-distilled-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_pipeline = pipeline("question-answering", model=model, tokenizer=tokenizer)

    # Тестируем с примером
    question = "What is AI?"
    context = "Artificial intelligence is the simulation of human intelligence in machines."
    result = qa_pipeline(question=question, context=context)
    
    assert "intelligence" in result['answer']
