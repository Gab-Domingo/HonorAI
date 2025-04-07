import streamlit as st
import os
import shutil
from document_processor import process_document, extract_document_text
from legal_ner import extract_legal_entities
from vector_store import create_document_embeddings, perform_document_search
from chatbot import get_chatbot_response
from database import setup_database, save_document, save_entities, save_chat_interaction, get_document_by_id, list_documents, delete_document
import utils

# Page configuration
st.set_page_config(
    page_title="Legal Document Analyzer",
    page_icon="⚖️",
    layout="wide"
)

# Cleanup temp directory at startup
temp_dir = "temp"
if os.path.exists(temp_dir):
    try:
        shutil.rmtree(temp_dir)
    except Exception as e:
        st.error(f"Error cleaning temp directory: {e}")
os.makedirs(temp_dir, exist_ok=True)

# Initialize database
try:
    setup_database()
except Exception as e:
    st.error(f"Error setting up database: {e}")

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
if "current_document_id" not in st.session_state:
    st.session_state.current_document_id = None

# Title and description
st.title("AI-Powered Legal Document Analyzer")
st.write("""
Upload legal documents to analyze key components, extract legal entities, 
and ask questions about the content using advanced AI technology.
""")

# Sidebar with tabs
tab1, tab2, tab3 = st.tabs(["📄 Document Analysis", "💬 Legal Assistant", "📚 Saved Documents"])

with tab1:
    # File uploader
    uploaded_file = st.file_uploader("Upload a legal document", type=["txt", "pdf", "docx"])
    
    # Document processing section
    if uploaded_file is not None:
        with st.spinner("Processing document..."):
            try:
                # Create a temporary file with explicit permissions
                # Make sure the temporary directory exists with proper permissions
                os.makedirs("temp", exist_ok=True)
                
                # Get file details for debugging
                file_details = {"FileName": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": uploaded_file.size}
                st.write(f"File details: {file_details}")
                
                # Save the file to our own temp directory instead of using tempfile
                temp_path = os.path.join("temp", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"File saved successfully to {temp_path}")
                
                # Extract text from document
                document_text = extract_document_text(temp_path)
                st.session_state.document_text = document_text
                
                st.info(f"Document text extracted successfully (length: {len(document_text)} characters)")
            except Exception as e:
                st.error(f"Error processing uploaded file: {e}")
                import traceback
                st.error(traceback.format_exc())
            
            # Process document with AI
            if st.button("Analyze Document"):
                try:
                    with st.spinner("Analyzing document with AI..."):
                        # Process document with LLM
                        analysis_results = process_document(document_text)
                        st.session_state.document_analysis = analysis_results
                        
                        # Extract entities
                        entities = extract_legal_entities(document_text)
                        st.session_state.entities = entities
                        
                        # Create embeddings for search
                        embeddings = create_document_embeddings(document_text)
                        st.session_state.document_embeddings = embeddings
                        
                        # Save to database
                        document_id = save_document(uploaded_file.name, document_text, analysis_results)
                        if document_id:
                            st.session_state.current_document_id = document_id
                            # Save entities to database
                            save_entities(document_id, entities)
                            st.success(f"Document saved to database with ID: {document_id}")
                        else:
                            st.error("Failed to save document to database")
                    
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
                except Exception as e:
                    st.error(f"Error analyzing document: {e}")

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
                
                # Save chat interaction to database if document is saved
                if st.session_state.current_document_id:
                    try:
                        save_chat_interaction(
                            st.session_state.current_document_id,
                            user_question,
                            response
                        )
                    except Exception as e:
                        st.error(f"Error saving chat interaction: {e}")
        
        # Display chat history
        chat_container = st.container()
        with chat_container:
            for message in st.session_state.chat_history:
                if message["role"] == "user":
                    st.markdown(f"**You:** {message['content']}")
                else:
                    st.markdown(f"**AI:** {message['content']}")
                st.markdown("---")

with tab3:
    st.subheader("Saved Documents")
    
    try:
        # Fetch saved documents from database
        documents = list_documents(limit=20)
        
        if not documents:
            st.info("No documents have been saved yet. Upload and analyze a document to save it.")
        else:
            st.write(f"Found {len(documents)} saved document(s):")
            
            # Create a table of documents
            for doc in documents:
                with st.expander(f"{doc['filename']} - {doc['document_type'] or 'Unknown Type'} - {doc['upload_date'].strftime('%Y-%m-%d %H:%M')}"):
                    st.write(f"**Summary:** {doc['summary'] or 'No summary available'}")
                    
                    # Load button
                    if st.button(f"Load Document #{doc['id']}", key=f"load_{doc['id']}"):
                        # Load document from database
                        full_doc = get_document_by_id(doc['id'])
                        if full_doc:
                            # Update session state
                            st.session_state.document_text = full_doc['document_text']
                            st.session_state.current_document_id = full_doc['id']
                            
                            # Reconstruct analysis results
                            analysis_results = {
                                "document_type": full_doc['document_type'],
                                "summary": full_doc['summary'],
                                "key_information": full_doc['key_information']
                            }
                            st.session_state.document_analysis = analysis_results
                            
                            # Set entities
                            entities = [
                                {
                                    "text": entity['entity_text'],
                                    "type": entity['entity_type'],
                                    "start": entity['start_pos'],
                                    "end": entity['end_pos']
                                } for entity in full_doc['entities']
                            ]
                            st.session_state.entities = entities
                            
                            # Create new embeddings for the document
                            embeddings = create_document_embeddings(full_doc['document_text'])
                            st.session_state.document_embeddings = embeddings
                            
                            st.success(f"Document #{doc['id']} loaded successfully!")
                            st.rerun()
                    
                    # Delete button
                    if st.button(f"Delete Document #{doc['id']}", key=f"delete_{doc['id']}"):
                        if delete_document(doc['id']):
                            st.success(f"Document #{doc['id']} deleted successfully!")
                            
                            # Clear session state if the current document is deleted
                            if st.session_state.current_document_id == doc['id']:
                                st.session_state.document_text = None
                                st.session_state.document_analysis = None
                                st.session_state.entities = None
                                st.session_state.document_embeddings = None
                                st.session_state.current_document_id = None
                                st.session_state.chat_history = []
                            
                            st.rerun()
    except Exception as e:
        st.error(f"Error loading documents from database: {e}")

# Footer
st.markdown("---")
st.markdown("Powered by OpenAI GPT-4o and LangChain")
