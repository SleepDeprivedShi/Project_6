import pdfplumber, docx, os

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

def _extract_docx(path: str) -> str:
    doc = docx.Document(path)
    paralist = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text:
            paralist.append(text)
    return "\n".join(paralist)



def dispatch_method(path: str) -> str:
    _, ext = os.path.splitext(path)
    ext = ext.lower()

    if ext == '.txt':
        return _extract_txt(path)
    elif ext == '.pdf':
        return _extract_pdf(path)
    elif ext == '.docx':
        return _extract_docx(path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")