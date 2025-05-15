import pytest
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

def load_model():
    model_name = "distilbert-base-uncased-distilled-squad"
    model = AutoModelForQuestionAnswering.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    qa_model = pipeline("question-answering", model=model, tokenizer=tokenizer)
    return qa_model

qa_pipeline = load_model()

def test_qa_pipeline_success():
    question = "What is AI?"
    context = "Artificial intelligence is the simulation of human intelligence in machines."
    result = qa_pipeline(question=question, context=context)
    assert "intelligence" in result['answer']

def test_qa_pipeline_empty_input():
    with pytest.raises(Exception):
        qa_pipeline(question="", context="")

def test_qa_pipeline_invalid_input():
    with pytest.raises(Exception):
        qa_pipeline(question=None, context=None)
