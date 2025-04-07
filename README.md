# Legal Document Analyzer

An AI-powered tool for analyzing legal documents, extracting key entities, and providing smart insights using advanced language models and natural language processing.

## Features

- **Document Processing**: Upload and analyze legal documents in multiple formats (PDF, DOCX, TXT)
- **AI-Powered Analysis**: Extract key information, summaries, and document type classification
- **Legal Entity Recognition**: Automatically identify and highlight legal entities such as:
  - Names
  - Organizations
  - Dates
  - Legal references
  - Monetary values
  - Contractual clauses
- **Interactive Chatbot**: Ask questions about your legal documents and receive context-aware answers
- **Document Management**: Save, retrieve, and manage your document analysis history
- **Seamless Database Integration**: PostgreSQL database for storing documents and analysis results

## Technology Stack

- **Frontend**: Streamlit for an interactive web interface
- **AI/ML**: OpenAI GPT-4o for advanced text analysis and understanding
- **NLP**: Custom Named Entity Recognition (NER) using SpaCy and LLM-powered entity extraction
- **Vector Search**: FAISS for efficient similarity search of document content
- **Document Processing**: Libraries for handling various document formats (PyPDF2, python-docx)
- **Database**: PostgreSQL for persistent storage

## Getting Started

### Prerequisites

- Python 3.8 or higher
- PostgreSQL database
- OpenAI API key

### Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/legal-document-analyzer.git
   cd legal-document-analyzer
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - `OPENAI_API_KEY`: Your OpenAI API key
   - `DATABASE_URL`: PostgreSQL connection string

4. Run the application:
   ```
   streamlit run app.py
   ```

## Usage

1. **Upload Document**: Use the file uploader to submit your legal document
2. **Analyze**: Click the "Analyze Document" button to process your document
3. **Review**: Explore the document summary, key information, and highlighted entities
4. **Ask Questions**: Use the chatbot to ask specific questions about your document
5. **Manage Documents**: Access your previously uploaded documents from the "Saved Documents" tab

## Implementation Details

- **Document Processing**: Multi-format support with text extraction and chunking
- **Retrieval-Augmented Generation (RAG)**: Enhances AI responses with context from the document
- **Named Entity Recognition**: Combines rule-based patterns and machine learning for entity extraction
- **Vector Embeddings**: Creates semantic search capability for relevant document sections

## Future Enhancements

- Support for more document formats
- Comparative analysis between multiple documents
- Legal precedent recommendations
- Enhanced entity visualization
- Template-based document generation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- OpenAI for GPT-4o API
- Streamlit for the web framework
- SpaCy for NLP capabilities# HonorAI
