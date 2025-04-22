from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.document import Document
from LLM_chains import load_vectordb, create_embeddings
import pypdfium2
import io


def extract_text_from_pdf(pdf_bytes):
    if isinstance(pdf_bytes, bytes):
        pdf_bytes = io.BytesIO(pdf_bytes)
    pdf_file = pypdfium2.PdfDocument(pdf_bytes)
    return "\n".join(
        pdf_file.get_page(i).get_textpage().get_text_range()
        for i in range(len(pdf_file))
    )


def get_pdf_text(pdf_input):
    if isinstance(pdf_input, list):
        return [extract_text_from_pdf(file) for file in pdf_input]
    else:
        return [extract_text_from_pdf(pdf_input)]

def get_text_chunks(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000, chunk_overlap=50, separators=["\n", "\n\n"]
    )
    return splitter.split_text(text)

def get_document_chunks(texts):
    documents = []
    for text in texts:
        chunks = get_text_chunks(text)
        for chunk in chunks:
            documents.append(Document(page_content=chunk))
    return documents

def add_documents_to_db(pdf_input):
    texts = get_pdf_text(pdf_input)
    documents = get_document_chunks(texts)
    vector_db = load_vectordb(create_embeddings())
    vector_db.add_documents(documents)
