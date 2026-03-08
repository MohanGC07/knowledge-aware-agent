import logging
from groq import Groq, APIError, AuthenticationError, RateLimitError
from config import GROQ_API_KEY, LLM_MODEL, LLM_MAX_TOKENS, LLM_TEMPERATURE

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """\
You are a precise, helpful AI assistant.
Answer the user's question using ONLY the context provided below.
If the answer cannot be found in the context, say:
"I couldn't find relevant information in the uploaded documents."
Do not speculate or use outside knowledge.\
"""

class Generator:

    def __init__(self) -> None:
        if not GROQ_API_KEY:
            raise EnvironmentError(
                "Groq API key is not set. Add GROQ_API_KEY to your .env or Streamlit secrets."
            )
        self._client = Groq(api_key=GROQ_API_KEY)

    def generate(self, query: str, context: str) -> str:
        if not query.strip():
            return "Please enter a question."
        if not context.strip():
            return "No relevant context was found in the uploaded documents."

        user_message = (
            "### Retrieved Context\n\n"
            f"{context}\n\n"
            "---\n\n"
            "### Question\n\n"
            f"{query}"
        )

        try:
            response = self._client.chat.completions.create(
                model=LLM_MODEL,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                max_tokens=LLM_MAX_TOKENS,
                temperature=LLM_TEMPERATURE,
            )
            return response.choices[0].message.content

        except AuthenticationError:
            return "Authentication failed. Please verify your Groq API key."
        except RateLimitError:
            return "The API rate limit has been reached. Please wait a moment and try again."
        except APIError as exc:
            logger.exception("Groq API error: %s", exc)
            return f"An API error occurred: {exc}"
        except Exception as exc:
            logger.exception("Unexpected error: %s", exc)
            return "An unexpected error occurred. Please try again."