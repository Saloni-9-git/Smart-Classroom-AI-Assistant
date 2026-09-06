# quiz.py

import json
from gemini_client import gemini


class QuizGenerator:
    """
    Professional quiz engine.
    """

    def generate_quiz(self, topic, difficulty):
        prompt = f"""
        Generate 5 high-quality {difficulty} level MCQs on {topic}.

        Rules:
        - Questions should test concepts, logic, and application
        - Avoid too basic questions
        - Make them exam-level
        - Each question must have:
            question
            options (A,B,C,D)
            correct_answer

        Return ONLY valid JSON list.

        Example:
        [
            {{
                "question": "...",
                "options": {{
                    "A": "...",
                    "B": "...",
                    "C": "...",
                    "D": "..."
                }},
                "correct_answer": "A"
            }}
        ]
        """

        response = gemini.generate(prompt)

        try:
            return json.loads(response)
        except:
            return []