from docx import Document


class WordHandler:

    @staticmethod
    def extract_text(file_path):

        doc = Document(file_path)

        text = ""

        for para in doc.paragraphs:
            text += para.text + "\n"

        return text