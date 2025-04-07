import os
from openai import OpenAI

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

def get_chatbot_response(question, context, document_analysis=None):
    """
    Get response from chatbot using OpenAI API with document context
    """
    # Prepare document information
    doc_info = ""
    if document_analysis:
        doc_type = document_analysis.get("document_type", "")
        key_info = document_analysis.get("key_information", {})
        
        if doc_type:
            import json
            doc_type_info = json.loads(doc_type) if isinstance(doc_type, str) else doc_type
            doc_info += f"Document Type: {doc_type_info.get('document_type', '')}\n"
            doc_info += f"Document Type Explanation: {doc_type_info.get('explanation', '')}\n\n"
        
        if key_info:
            doc_info += "Document Key Information:\n"
            for key, value in key_info.items():
                if isinstance(value, str):
                    doc_info += f"- {key}: {value}\n"
                elif isinstance(value, dict):
                    doc_info += f"- {key}:\n"
                    for sub_key, sub_value in value.items():
                        doc_info += f"  - {sub_key}: {sub_value}\n"
    
    # Construct prompt with context
    prompt = f"""
    You are a legal assistant specializing in document analysis and legal research.
    
    Document Context:
    {context}
    
    {doc_info}
    
    User Question: {question}
    
    Please provide a detailed and accurate response to the user's question based on the document context.
    If the answer cannot be determined from the provided context, acknowledge this limitation and provide
    general legal information that might still be helpful. Cite specific sections or clauses from the
    document when relevant.
    """
    
    # Get response from OpenAI
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": "You are a legal assistant specializing in document analysis and legal research. Provide clear, accurate and helpful responses to questions about legal documents."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.2,
        max_tokens=1000
    )
    
    return response.choices[0].message.content

def get_legal_information(query):
    """
    Get general legal information for questions that can't be answered from the document
    """
    prompt = f"""
    You are a legal assistant with expertise in various legal domains. 
    
    User Query: {query}
    
    Please provide helpful general legal information related to this query. 
    Make it clear that this is general information and not specific legal advice.
    Include a disclaimer about consulting a qualified attorney for specific legal matters.
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=800
    )
    
    return response.choices[0].message.content
