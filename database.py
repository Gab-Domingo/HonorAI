import os
import json
import psycopg2
from psycopg2.extras import RealDictCursor
from datetime import datetime

# Get database connection details from environment variables
def get_db_connection():
    """Create a connection to the PostgreSQL database"""
    conn = psycopg2.connect(
        host=os.environ.get("PGHOST"),
        database=os.environ.get("PGDATABASE"),
        user=os.environ.get("PGUSER"),
        password=os.environ.get("PGPASSWORD"),
        port=os.environ.get("PGPORT")
    )
    return conn

def setup_database():
    """Create necessary tables if they don't exist"""
    conn = get_db_connection()
    cur = conn.cursor()
    try:
        print("Setting up database tables...")
        
        # Create documents table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS documents (
                id SERIAL PRIMARY KEY,
                filename TEXT NOT NULL,
                document_type TEXT,
                upload_date TIMESTAMP NOT NULL,
                document_text TEXT NOT NULL,
                summary TEXT,
                key_information JSONB
            )
        ''')
        print("Created documents table")
        
        # Create entities table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS entities (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL,
                entity_text TEXT NOT NULL,
                entity_type TEXT NOT NULL,
                start_pos INTEGER,
                end_pos INTEGER,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        ''')
        print("Created entities table")
        
        # Create chat_history table
        cur.execute('''
            CREATE TABLE IF NOT EXISTS chat_history (
                id SERIAL PRIMARY KEY,
                document_id INTEGER NOT NULL,
                user_query TEXT NOT NULL,
                assistant_response TEXT NOT NULL,
                timestamp TIMESTAMP NOT NULL,
                FOREIGN KEY (document_id) REFERENCES documents (id) ON DELETE CASCADE
            )
        ''')
        print("Created chat_history table")
        
        conn.commit()
        print("Database setup completed successfully")
    except Exception as e:
        conn.rollback()
        print(f"Error setting up database: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def save_document(filename, document_text, document_analysis=None):
    """Save document and its analysis to the database"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Extract document analysis data if available
        document_type = None
        summary = None
        key_information = None
        
        if document_analysis:
            document_type = document_analysis.get("document_type")
            summary = document_analysis.get("summary")
            key_information = json.dumps(document_analysis.get("key_information", {}))
        
        # Insert document
        cur.execute('''
            INSERT INTO documents 
            (filename, document_type, upload_date, document_text, summary, key_information)
            VALUES (%s, %s, %s, %s, %s, %s)
            RETURNING id
        ''', (filename, document_type, datetime.now(), document_text, summary, key_information))
        
        result = cur.fetchone()
        document_id = result['id'] if result else None
        conn.commit()
        return document_id
    except Exception as e:
        conn.rollback()
        print(f"Error saving document: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def save_entities(document_id, entities):
    """Save extracted entities to the database"""
    if not entities:
        return
    
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        for entity in entities:
            cur.execute('''
                INSERT INTO entities 
                (document_id, entity_text, entity_type, start_pos, end_pos)
                VALUES (%s, %s, %s, %s, %s)
            ''', (
                document_id, 
                entity["text"], 
                entity["type"], 
                entity.get("start", 0), 
                entity.get("end", 0)
            ))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving entities: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def save_chat_interaction(document_id, user_query, assistant_response):
    """Save a chat interaction to the database"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        cur.execute('''
            INSERT INTO chat_history 
            (document_id, user_query, assistant_response, timestamp)
            VALUES (%s, %s, %s, %s)
        ''', (document_id, user_query, assistant_response, datetime.now()))
        
        conn.commit()
    except Exception as e:
        conn.rollback()
        print(f"Error saving chat interaction: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def get_document_by_id(document_id):
    """Retrieve a document and its analysis by ID"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        # Get document
        cur.execute('''
            SELECT * FROM documents WHERE id = %s
        ''', (document_id,))
        
        document = cur.fetchone()
        
        if not document:
            return None
        
        # Parse JSON data
        if document.get('key_information'):
            document['key_information'] = json.loads(document['key_information'])
        else:
            document['key_information'] = {}
            
        # Get entities
        cur.execute('''
            SELECT * FROM entities WHERE document_id = %s
        ''', (document_id,))
        
        entities = cur.fetchall()
        document['entities'] = entities
        
        return document
    except Exception as e:
        print(f"Error retrieving document: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def get_chat_history(document_id, limit=10):
    """Retrieve chat history for a document"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute('''
            SELECT * FROM chat_history 
            WHERE document_id = %s
            ORDER BY timestamp DESC
            LIMIT %s
        ''', (document_id, limit))
        
        history = cur.fetchall()
        return history
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def list_documents(limit=10):
    """List all documents in the database"""
    conn = get_db_connection()
    cur = conn.cursor(cursor_factory=RealDictCursor)
    
    try:
        cur.execute('''
            SELECT id, filename, document_type, upload_date, summary 
            FROM documents
            ORDER BY upload_date DESC
            LIMIT %s
        ''', (limit,))
        
        documents = cur.fetchall()
        return documents
    except Exception as e:
        print(f"Error listing documents: {e}")
        raise
    finally:
        cur.close()
        conn.close()

def delete_document(document_id):
    """Delete a document and all related data"""
    conn = get_db_connection()
    cur = conn.cursor()
    
    try:
        # Due to CASCADE constraint, this will delete related entities and chat history
        cur.execute('''
            DELETE FROM documents WHERE id = %s
        ''', (document_id,))
        
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        print(f"Error deleting document: {e}")
        raise
    finally:
        cur.close()
        conn.close()

# Initialize database tables when module is imported
try:
    print("Initializing database on module load...")
    setup_database()
    print("Database initialized successfully!")
except Exception as e:
    print(f"Error initializing database on module load: {e}")