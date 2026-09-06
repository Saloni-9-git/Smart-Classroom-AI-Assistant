# summarizer.py
from gemini_client import gemini

class SmartSummarizer:
    """
    AI Smart Notes Generator
    """

    def generate(self, prompt):
        """
        Send any prompt to the LLM.
        """
        return gemini.generate(prompt)

    def summarize_chunk(self, chunk, note_type):
        """
        Summarize one chunk according to note type.
        """

        if note_type == "Quick Revision":

            prompt = f"""
Create quick revision notes.

Rules:
- Bullet points
- Important keywords
- Important formulas
- Important definitions
- Maximum one page

Text:
{chunk}
"""

        elif note_type == "Standard Notes":

            prompt = f"""
Create professional classroom notes.

Include:
- Headings
- Concepts
- Definitions
- Examples
- Important formulas
- Key points

Text:
{chunk}
"""

        else:

            prompt = f"""
Explain this like a teacher teaching a beginner.

Rules:
- Very easy English
- Explain every difficult term
- Give examples
- Step-by-step explanation

Text:
{chunk}
"""

        return gemini.generate(prompt)

    def summarize_large_text(self, chunks, note_type):

        summaries = []

        # Summarize each chunk separately
        for chunk in chunks:

            summaries.append(
                self.summarize_chunk(chunk, note_type)
            )

        merged = "\n".join(summaries)

        final_prompt = f"""
Combine all these notes into one clean document.

Remove duplicate points.

Organize properly.

Notes:

{merged}
"""

        return gemini.generate(final_prompt)