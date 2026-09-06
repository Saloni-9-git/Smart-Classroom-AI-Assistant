import faiss
import numpy as np

from sentence_transformers import SentenceTransformer

from pdf_handler import PDFHandler
from loaders.word_loader import WordHandler
from loaders.excel_loader import ExcelHandler
from loaders.image_loader import OCRHandler
from utils import TextProcessor
from gemini_client import gemini


class SmartRAG:

    def __init__(self):

        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.index = None
        self.text_chunks = []

    # ============================================================
    # PROCESS UPLOADED FILE
    # ============================================================

    def process_file(self, file_path):

        extension = file_path.split(".")[-1].lower()

        try:

            if extension == "pdf":
                text = PDFHandler.extract_text(file_path)

            elif extension == "docx":
                text = WordHandler.extract_text(file_path)

            elif extension in ["xlsx", "xls"]:
                text = ExcelHandler.extract_text(file_path)

            elif extension in ["png", "jpg", "jpeg"]:
                text = OCRHandler.extract_text(file_path)

            else:
                return

            if not text or not text.strip():
                return

            chunks = TextProcessor.split_text(text)

            chunks = [
                chunk.strip()
                for chunk in chunks
                if chunk and chunk.strip()
            ]

            if not chunks:
                return

            self.text_chunks.extend(chunks)

            embeddings = self.model.encode(
                chunks,
                normalize_embeddings=True
            )

            embeddings = np.asarray(
                embeddings,
                dtype="float32"
            )

            if self.index is None:

                self.index = faiss.IndexFlatIP(
                    embeddings.shape[1]
                )

            self.index.add(embeddings)

        except Exception as e:

            print(
                f"Error processing file {file_path}: {e}"
            )

    # ============================================================
    # RETRIEVE RELEVANT CONTEXT
    # ============================================================

    def retrieve(
        self,
        query,
        top_k=5,
        similarity_threshold=0.30
    ):

        if self.index is None:
            return []

        if not self.text_chunks:
            return []

        try:

            query_embedding = self.model.encode(
                [query],
                normalize_embeddings=True
            )

            query_embedding = np.asarray(
                query_embedding,
                dtype="float32"
            )

            actual_k = min(
                top_k,
                len(self.text_chunks)
            )

            similarities, indices = self.index.search(
                query_embedding,
                actual_k
            )

            results = []

            for similarity, idx in zip(
                similarities[0],
                indices[0]
            ):

                if idx < 0 or idx >= len(self.text_chunks):
                    continue

                if float(similarity) < similarity_threshold:
                    continue

                results.append(
                    {
                        "text": self.text_chunks[idx],
                        "score": float(similarity)
                    }
                )

            return results

        except Exception as e:

            print(
                f"Retrieval error: {e}"
            )

            return []

    # ============================================================
    # CONVERSATION CONTEXT
    # ============================================================

    def _build_conversation_context(
        self,
        chat_history,
        max_messages=6
    ):

        if not chat_history:
            return ""

        recent = chat_history[-max_messages:]

        parts = []

        for message in recent:

            role = message.get("role")

            content = message.get(
                "content",
                ""
            ).strip()

            if not content:
                continue

            if role == "user":

                parts.append(
                    f"Student: {content}"
                )

            elif role == "assistant":

                parts.append(
                    f"Assistant: {content}"
                )

        return "\n".join(parts)

    # ============================================================
    # CREATE SEARCH QUERY FOR FOLLOW-UP QUESTIONS
    # ============================================================

    def _build_search_query(
        self,
        question,
        chat_history
    ):

        if not chat_history:
            return question

        recent_user_questions = []

        for message in reversed(chat_history):

            if message.get("role") == "user":

                content = message.get(
                    "content",
                    ""
                ).strip()

                if content:
                    recent_user_questions.append(
                        content
                    )

                if len(recent_user_questions) >= 2:
                    break

        recent_user_questions.reverse()

        if not recent_user_questions:
            return question

        return (
            "Previous related questions:\n"
            + "\n".join(recent_user_questions)
            + "\nCurrent question:\n"
            + question
        )

    # ============================================================
    # BUILD SOURCE CONTEXT
    # ============================================================

    def _build_context(
        self,
        retrieved
    ):

        if not retrieved:
            return ""

        context_parts = []

        for i, item in enumerate(
            retrieved,
            start=1
        ):

            context_parts.append(
                f"""
SOURCE {i}
{item["text"]}
"""
            )

        return "\n".join(
            context_parts
        )

    # ============================================================
    # CHECK WHETHER MATERIAL SUPPORTS QUESTION
    # ============================================================

    def _check_material_support(
        self,
        question,
        context,
        conversation_context=""
    ):

        if not context.strip():
            return False

        prompt = f"""
You are checking whether a student's current question
can be answered from the uploaded study material.

Uploaded Study Material:
========================
{context}
========================

Recent Conversation:
====================
{conversation_context}
====================

Current Student Question:
{question}

Important:

- Check ONLY the uploaded study material.
- Do not use general knowledge.
- The answer does not need to use the exact same words.
- If the material clearly contains the concept or enough
  information to answer the question, reply YES.
- If the material does not contain enough information,
  reply NO.
- For follow-up questions such as "why?", "explain this",
  "what does that mean?", use the recent conversation to
  understand what "this" refers to, but still check whether
  the actual explanation is supported by the uploaded material.

Reply with ONLY:
YES
or
NO
"""

        try:

            result = gemini.generate(
                prompt
            ).strip().upper()

            return result.startswith("YES")

        except Exception as e:

            print(
                f"Material verification error: {e}"
            )

            return False

    # ============================================================
    # ANSWER FROM MATERIAL
    # ============================================================

    def _answer_from_material(
        self,
        question,
        context,
        conversation_context=""
    ):

        prompt = f"""
You are Smart Classroom AI, a friendly teacher.

Answer the student's CURRENT question using ONLY the
uploaded study material.

Uploaded Study Material:
========================
{context}
========================

Recent Conversation:
====================
{conversation_context}
====================

Current Question:
{question}

Rules:

1. Answer the CURRENT question directly.

2. Use the recent conversation only to understand
   references such as "this", "that", "why", or "explain it".

3. Do not add facts that are not supported by the
   uploaded material.

4. Use EASY TO MEDIUM English.

5. Make the answer understandable to intelligent,
   average and weaker students.

6. Avoid research-paper language.

7. Explain difficult terms in simple words.

8. Use short paragraphs or bullet points.

9. Do not unnecessarily expand a simple answer.

10. Keep the same general difficulty level as the
    uploaded material.

11. If the material contains an example, use it when
    it helps the student understand.

12. Do not invent examples, definitions or formulas.

13. The goal is understanding, not impressive vocabulary.

Give the answer now.
"""

        return gemini.generate(
            prompt
        )

    # ============================================================
    # GENERAL KNOWLEDGE FALLBACK
    # ============================================================

    def _answer_from_general_knowledge(
        self,
        question,
        conversation_context=""
    ):

        prompt = f"""
You are Smart Classroom AI, a friendly teacher.

The uploaded study material does NOT contain enough
information to answer the student's current question.

First write exactly:

This information is not available in your uploaded material.

Then write:

However, here is a general explanation:

Then answer the CURRENT question using your general knowledge.

Recent Conversation:
====================
{conversation_context}
====================

Current Question:
{question}

Rules:

- Use EASY TO MEDIUM English.
- Make it understandable to intelligent, average and weaker students.
- Use simple explanations.
- Give the direct answer first.
- Explain technical terms simply.
- Use short paragraphs or bullet points.
- Do not make the answer unnecessarily long.
- Do not pretend that the answer came from the PDF.
- Do not invent citations or claim that you searched the web.
- Use the recent conversation only to understand references
  such as "this", "that", "why", or "explain more".

Give the answer now.
"""

        return gemini.generate(
            prompt
        )

    # ============================================================
    # MAIN CHATBOT
    # ============================================================

    def ask(
        self,
        question,
        chat_history=None
    ):

        question = question.strip()

        chat_history = (
            chat_history
            if chat_history
            else []
        )

        if not question:

            return {
                "answer": "Please enter a question.",
                "sources": []
            }

        conversation_context = (
            self._build_conversation_context(
                chat_history
            )
        )

        search_query = (
            self._build_search_query(
                question,
                chat_history
            )
        )

        # --------------------------------------------------------
        # NO UPLOADED MATERIAL
        # --------------------------------------------------------

        if (
            self.index is None
            or not self.text_chunks
        ):

            answer = (
                self._answer_from_general_knowledge(
                    question,
                    conversation_context
                )
            )

            return {
                "answer": answer,
                "sources": []
            }

        # --------------------------------------------------------
        # RETRIEVE MATERIAL
        # --------------------------------------------------------

        retrieved = self.retrieve(
            search_query,
            top_k=5,
            similarity_threshold=0.30
        )

        # --------------------------------------------------------
        # NOTHING RELEVANT FOUND
        # --------------------------------------------------------

        if not retrieved:

            answer = (
                self._answer_from_general_knowledge(
                    question,
                    conversation_context
                )
            )

            return {
                "answer": answer,
                "sources": []
            }

        context = self._build_context(
            retrieved
        )

        # --------------------------------------------------------
        # CHECK PDF SUPPORT
        # --------------------------------------------------------

        material_supports_answer = (
            self._check_material_support(
                question,
                context,
                conversation_context
            )
        )

        # --------------------------------------------------------
        # PDF ANSWER
        # --------------------------------------------------------

        if material_supports_answer:

            answer = (
                self._answer_from_material(
                    question,
                    context,
                    conversation_context
                )
            )

            sources = [
                item["text"]
                for item in retrieved
            ]

            return {
                "answer": answer,
                "sources": sources
            }

        # --------------------------------------------------------
        # GENERAL KNOWLEDGE ANSWER
        # --------------------------------------------------------

        answer = (
            self._answer_from_general_knowledge(
                question,
                conversation_context
            )
        )

        return {
            "answer": answer,
            "sources": []
        }
