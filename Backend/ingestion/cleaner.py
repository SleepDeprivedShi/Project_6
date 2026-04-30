import re


def normalize_whitespace(text: str) -> str:
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'^[ \t]+|[ \t]+$', '', text, flags=re.MULTILINE)
    return text


def remove_non_printable(text: str) -> str:
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    return text


def remove_blank_pages(text: str) -> str:
    lines = text.split('\n')
    non_empty_lines = [line for line in lines if line.strip()]
    return '\n'.join(non_empty_lines)


def process(text: str) -> str:
    text = remove_non_printable(text)
    text = normalize_whitespace(text)
    text = remove_blank_pages(text)
    return text.strip()