import os
from pypdf import PdfReader
import docx


class DocumentLoader:

    def load_file(self, file_path):

        extension = file_path.split(".")[-1].lower()

        if extension == "pdf":
            return self.load_pdf(file_path)

        elif extension == "docx":
            return self.load_docx(file_path)

        elif extension in ["txt", "md", "py", "json", "csv"]:
            return self.load_text(file_path)

        else:
            return ""


    def load_pdf(self, path):

        reader = PdfReader(path)

        text = ""

        for page in reader.pages:
            content = page.extract_text()

            if content:
                text += content

        return text


    def load_docx(self, path):

        doc = docx.Document(path)

        text = ""

        for para in doc.paragraphs:
            text += para.text + "\n"

        return text


    def load_text(self, path):

        with open(path, "r", encoding="utf-8") as f:
            return f.read()