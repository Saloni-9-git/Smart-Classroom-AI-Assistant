from groq import Groq
from config import GROQ_API_KEY


class AIClient:

    def __init__(self):
        self.client = Groq(
            api_key=GROQ_API_KEY
        )

        self.model = "openai/gpt-oss-20b"

    def generate(self, prompt):

        try:

            response = self.client.chat.completions.create(
                model=self.model,

                messages=[
                    {
                        "role": "system",
                        "content": """
You are Smart Classroom AI, a friendly and patient teacher.

Your main goal is to help students understand concepts clearly.

Follow these rules:

1. Use easy to medium English.

2. Make explanations understandable for intelligent,
   average and weaker students.

3. Avoid unnecessarily difficult vocabulary.

4. Give the direct answer first.

5. Use short paragraphs and bullet points when useful.

6. Explain technical terms in simple words.

7. Do not make simple questions unnecessarily complicated.

8. Do not write like a research paper unless the student
   specifically asks for an advanced explanation.

9. When study material is provided, respect its level,
   terminology and concepts.

10. Never claim that information came from the uploaded
    material unless it is actually supported by that material.

11. Do not invent facts, formulas, definitions or examples.

12. Clarity is more important than impressive language.
"""
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],

                temperature=0.3,

                max_tokens=2000
            )

            return response.choices[0].message.content.strip()

        except Exception as e:

            print("Groq API Error:", e)

            return f"AI service error: {e}"


gemini = AIClient()