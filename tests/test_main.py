
import pytest
from main import model

def test_model_response():
    question = "Что такое искусственный интеллект?"
    response = model([question])[0]
    assert isinstance(response, str)
    assert len(response) > 0
    