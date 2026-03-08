import os
from groq import Groq
from config import LLM_MODEL
from config import GROQ_API_KEY
from dotenv import load_dotenv

load_dotenv()


class Generator:

    def __init__(self):

        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))

    def generate(self, query, context):

        prompt = f"""
Use the context below to answer the question.

Context:
{context}

Question:
{query}
"""

        response = self.client.chat.completions.create(
            model=LLM_MODEL,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )

        return response.choices[0].message.content