import pdfplumber, docx

def _extract_txt(path: str) -> str:
    with open(path, 'r', encoding = 'utf-8', errors= "ignore") as f:
        return f.read()

def _extract_pdf(path: str) -> str:
    text =''
    with pdfplumber.open(path) as pdf:
        for page in pdf.pages:
            try:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
            except Exception as e:
                print("Pdf had an error: ", e)
    return text
