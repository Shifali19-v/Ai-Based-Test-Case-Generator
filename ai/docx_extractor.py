from docx import Document


class DOCXExtractor:

    def extract_text(self, file_path):
        try:
            document = Document(file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"DOCX file not found: {file_path}")
        except Exception as e:
            raise ValueError(f"Unable to read DOCX file: {e}")

        paragraphs = []

        for paragraph in document.paragraphs:
            text = paragraph.text.strip()

            if text:
                paragraphs.append(text)

        if not paragraphs:
            raise ValueError("No readable text found in DOCX file")

        return "\n".join(paragraphs)
