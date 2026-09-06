from PIL import Image
import pytesseract


class OCRHandler:

    @staticmethod
    def extract_text(file_path):

        image = Image.open(file_path)

        text = pytesseract.image_to_string(image)

        return text