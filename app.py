import streamlit as st
import tempfile
import os
from document_processor import process_document, extract_document_text
from legal_ner import extract_legal_entities
from vector_store import create_document_embeddings, perform_document_search
from chatbot import get_chatbot_response
import utils

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Initialize session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "document_text" not in st.session_state:
    st.session_state.document_text = None
if "document_analysis" not in st.session_state:
    st.session_state.document_analysis = None
if "entities" not in st.session_state:
    st.session_state.entities = None
if "document_embeddings" not in st.session_state:
    st.session_state.document_embeddings = None

# Title and description
st.title("AI-Powered Legal Document Analyzer")
st.write("""
Upload legal documents to analyze key components, extract legal entities, 
and ask questions about the content using advanced AI technology.
""")

# Sidebar with tabs
tab1, tab2 = st.tabs(["📄 Document Analysis", "💬 Legal Assistant"])

with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "pdf", "docx"])
    
    # Document processing section
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            # Save uploaded file to temp file
            temp_dir = tempfile.TemporaryDirectory()
            temp_path = os.path.join(temp_dir.name, uploaded_file.name)
            with open(temp_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Extract text from document
            document_text = extract_document_text(temp_path)
            st.session_state.document_text = document_text
            
            # Process document with AI
            if st.button("Analyze Document"):
                # Process document with LLM
                analysis_results = process_document(document_text)
                st.session_state.document_analysis = analysis_results
                
                # Extract entities
                entities = extract_legal_entities(document_text)
                st.session_state.entities = entities
                
                # Create embeddings for search
                embeddings = create_document_embeddings(document_text)
                st.session_state.document_embeddings = embeddings
                
                # Display results
                st.subheader("Document Analysis")
                st.write(analysis_results["summary"])
                
                # Display document type and key information
                st.subheader("Document Type")
                st.write(analysis_results["document_type"])
                
                st.subheader("Key Information")
                for key, value in analysis_results["key_information"].items():
                    st.write(f"**{key}:** {value}")
                
                # Display extracted entities
                st.subheader("Legal Entities")
                entity_columns = st.columns(3)
                
                entity_types = {}
                for entity in entities:
                    entity_type = entity["type"]
                    if entity_type not in entity_types:
                        entity_types[entity_type] = []
                    entity_types[entity_type].append(entity["text"])
                
                for i, (entity_type, entity_values) in enumerate(entity_types.items()):
                    col_idx = i % 3
                    with entity_columns[col_idx]:
                        st.write(f"**{entity_type}**")
                        for value in entity_values:
                            st.write(f"- {value}")
                
                # Display highlighted document
                st.subheader("Document with Highlighted Entities")
                highlighted_text = utils.highlight_entities_in_text(document_text, entities)
                st.markdown(highlighted_text, unsafe_allow_html=True)

with tab2:
    # Check if document is uploaded
    if st.session_state.document_text is None:
        st.warning("Please upload and analyze a document first to use the chatbot.")
    else:
        st.subheader("Ask questions about your legal document")
        
        # Chat input
        user_question = st.text_input("Your question about the document:")
        
        if user_question and user_question.strip():
            # Add user question to chat history
            st.session_state.chat_history.append({"role": "user", "content": user_question})
            
            # Get response from chatbot
            with st.spinner("Generating response..."):
                if st.session_state.document_embeddings:
                    # Get relevant document context
                    context = perform_document_search(
                        user_question, 
                        st.session_state.document_embeddings, 
                        st.session_state.document_text
                    )
                    
                    # Get response from chatbot with context
                    response = get_chatbot_response(
                        user_question, 
                        context,
                        st.session_state.document_analysis
                    )
                else:
                    # Get response from chatbot without context
                    response = get_chatbot_response(
                        user_question, 
                        st.session_state.document_text,
                        st.session_state.document_analysis
                    )
                
                # Add response to chat history
                st.session_state.chat_history.append({"role": "assistant", "content": response})
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
                st.markdown("---")

# Footer
st.markdown("---")
st.markdown("Powered by OpenAI GPT-4o and LangChain")
