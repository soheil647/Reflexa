# tests/test_llm.py
from modules.llm.interaction import LLMInteraction

def test_llm_interaction():
    llm = LLMInteraction()
    user_profile = {"name": "TestUser"}
    metrics = {"heart_rate": 75, "skin_health": "good"}
    message = llm.get_recommendation(user_profile, metrics)
    assert isinstance(message, str) and len(message) > 0, "LLM did not return a valid message"
