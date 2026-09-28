import csv


class CSVExtractor:

    def extract_text(self, file_path):
        try:
            with open(file_path, "r", encoding="utf-8-sig", newline="") as file:
                reader = csv.reader(file)
                rows = list(reader)

        except FileNotFoundError:
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        except Exception as e:
            raise ValueError(f"Unable to read CSV file: {e}")

        if not rows:
            raise ValueError("No readable data found in CSV file")

        text_rows = []

        for row in rows:
            values = [value.strip() for value in row if value.strip()]

            if values:
                text_rows.append(" | ".join(values))

        if not text_rows:
            raise ValueError("No readable data found in CSV file")

        return "\n".join(text_rows)
