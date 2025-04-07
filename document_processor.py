import os
import re
import PyPDF2
import docx
from langchain.text_splitter import RecursiveCharacterTextSplitter
from openai import OpenAI

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def extract_document_text(file_path):
    """
    Extract text from various document formats
    """
    file_extension = file_path.split('.')[-1].lower()
    
    if file_extension == 'pdf':
        return extract_text_from_pdf(file_path)
    elif file_extension == 'docx':
        return extract_text_from_docx(file_path)
    elif file_extension == 'txt':
        return extract_text_from_txt(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_extension}")

def extract_text_from_pdf(file_path):
    """Extract text from PDF file"""
    text = ""
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def extract_text_from_docx(file_path):
    """Extract text from DOCX file"""
    doc = docx.Document(file_path)
    return "\n".join([paragraph.text for paragraph in doc.paragraphs])

def extract_text_from_txt(file_path):
    """Extract text from TXT file"""
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def split_text_into_chunks(text, chunk_size=4000, overlap=200):
    """
    Split document text into manageable chunks for processing
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        length_function=len
    )
    return text_splitter.split_text(text)

def process_document(document_text):
    """
    Process document using OpenAI GPT model to extract key information
    """
    # Split text if it's too long
    if len(document_text) > 8000:
        # Get a summary first to reduce token usage
        summary = get_document_summary(document_text)
        chunks = split_text_into_chunks(document_text)
        doc_type = determine_document_type(summary)
        key_info = extract_key_information(chunks, doc_type)
    else:
        # Process the whole document
        summary = get_document_summary(document_text)
        doc_type = determine_document_type(document_text)
        key_info = extract_key_information([document_text], doc_type)
    
    return {
        "summary": summary,
        "document_type": doc_type,
        "key_information": key_info
    }

def get_document_summary(document_text):
    """
    Generate a summary of the document using OpenAI
    """
    # Limit text length for the prompt
    max_chars = 14000  # Limit to leave room for prompt and completion
    truncated_text = document_text[:max_chars] if len(document_text) > max_chars else document_text
    
    prompt = f"""
    Please provide a comprehensive summary of the following legal document. 
    Include the main purpose, key points, and any notable provisions or clauses:
    
    {truncated_text}
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.3
    )
    
    return response.choices[0].message.content

def determine_document_type(document_text):
    """
    Determine the type of legal document using OpenAI
    """
    # Limit text length for the prompt
    max_chars = 12000  # Limit to leave room for prompt and completion
    truncated_text = document_text[:max_chars] if len(document_text) > max_chars else document_text
    
    prompt = f"""
    Based on the following legal document, identify the specific type of legal document it is 
    (e.g., contract, NDA, employment agreement, terms of service, privacy policy, patent, 
    trademark registration, court filing, etc.). 
    
    Please respond with a concise document type label and a short explanation of what indicates this document type:
    
    {truncated_text}
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    return result

def extract_key_information(document_chunks, document_type):
    """
    Extract key information from document based on its type
    """
    import json
    
    doc_type_obj = json.loads(document_type)
    doc_type_label = doc_type_obj.get("document_type", "Legal Document")
    
    # Process first chunk as it usually contains the most important information
    first_chunk = document_chunks[0]
    max_chars = 12000  # Limit to leave room for prompt and completion
    truncated_text = first_chunk[:max_chars] if len(first_chunk) > max_chars else first_chunk
    
    prompt = f"""
    You are analyzing a {doc_type_label}. 
    Based on the following document excerpt, extract the key information relevant to this type of legal document.
    
    For example, for contracts, extract parties, dates, values, jurisdiction, etc.
    For court filings, extract case number, parties, court, filing date, etc.
    For patents/trademarks, extract filing numbers, owners, dates, descriptions, etc.
    
    Document excerpt:
    {truncated_text}
    
    Extract and organize the key information as JSON with appropriate fields for this document type.
    Only include fields where you find specific information in the document.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    result = json.loads(response.choices[0].message.content)
    return result
