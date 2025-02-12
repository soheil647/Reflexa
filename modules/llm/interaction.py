# modules/llm/interaction.py
import openai
from modules.utils.logger import get_logger
from config import LLM_API_KEY, LLM_ENGINE

logger = get_logger(__name__)

class LLMInteraction:
    def __init__(self):
        # Initialize OpenAI API key (ensure you securely manage your API key)
        openai.api_key = LLM_API_KEY

    def get_recommendation(self, user_profile, metrics):
        prompt = (
            f"User {user_profile['name']} has the following health metrics: {metrics}.\n"
            "Provide a concise recommendation for their wellness."
        )
        try:
            response = openai.Completion.create(
                engine=LLM_ENGINE,
                prompt=prompt,
                max_tokens=50,
                n=1,
                stop=None,
                temperature=0.7,
            )
            message = response.choices[0].text.strip()
            logger.info(f"LLM recommendation: {message}")
            return message
        except Exception as e:
            logger.error(f"Error in LLM interaction: {e}")
            return "I'm having trouble analyzing your health metrics right now."
