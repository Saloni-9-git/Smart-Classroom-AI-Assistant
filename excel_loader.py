import pandas as pd


class ExcelHandler:

    @staticmethod
    def extract_text(file_path):

        excel = pd.read_excel(file_path)

        return excel.to_string(index=False)