from pypdf import PdfReader

#Load PDFs
def load_data(paths):
    all_documents = []
    for path in paths:
        reader = PdfReader(path)
        text = ""
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        all_documents.append(text.strip())
    return all_documents

#Text Splitter
def split_text_with_overlap(text, chunk_size=1500, overlap=200):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap
    return chunks

#Prompt Function
def build_prompt(context: str, question: str) -> str:
    return f"""Use the following context to answer the question:

{context}

Question: {question}
"""