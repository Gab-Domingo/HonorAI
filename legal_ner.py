import os
import spacy
import re
from openai import OpenAI

# Initialize OpenAI client
# the newest OpenAI model is "gpt-4o" which was released May 13, 2024.
# do not change this unless explicitly requested by the user
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
openai = OpenAI(api_key=OPENAI_API_KEY)

# Try to load SpaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    # If not installed, use a smaller model or just initialize
    nlp = spacy.blank("en")

# Define legal entity types
LEGAL_ENTITY_TYPES = [
    "PERSON", "ORG", "GPE", "DATE", "MONEY", "LAW", "COURT", "JUDGE", "STATUTE",
    "REGULATION", "CASE_CITATION", "LEGAL_TERM", "PARTY", "JURISDICTION"
]

# Regex patterns for legal entities
legal_patterns = {
    "CASE_CITATION": r'([A-Za-z\s]+\sv\.\s[A-Za-z\s]+)|(\d+\s[A-Za-z\.]+\s\d+)',
    "STATUTE": r'([A-Za-z\.]+\s§\s\d+(?:\.\d+)?)',
    "MONEY": r'\$\d+(?:,\d+)*(?:\.\d+)?',
    "DATE": r'\d{1,2}/\d{1,2}/\d{2,4}|\d{1,2}-\d{1,2}-\d{2,4}|(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]* \d{1,2},? \d{4}',
}

def extract_legal_entities(text):
    """
    Extract legal entities from text using both SpaCy and custom patterns
    """
    entities = []
    
    # Use SpaCy for basic NER
    doc = nlp(text)
    for ent in doc.ents:
        if ent.label_ in ["PERSON", "ORG", "GPE", "DATE", "MONEY"]:
            entities.append({
                "text": ent.text,
                "start": ent.start_char,
                "end": ent.end_char,
                "type": ent.label_
            })
    
    # Use regex patterns for specific legal entities
    for entity_type, pattern in legal_patterns.items():
        for match in re.finditer(pattern, text):
            entities.append({
                "text": match.group(),
                "start": match.start(),
                "end": match.end(),
                "type": entity_type
            })
    
    # Use OpenAI for advanced legal entity recognition
    openai_entities = extract_legal_entities_with_llm(text)
    
    # Combine all entities and remove duplicates
    all_entities = entities + openai_entities
    unique_entities = remove_duplicate_entities(all_entities)
    
    return unique_entities

def extract_legal_entities_with_llm(text):
    """
    Use OpenAI to extract legal entities that might be missed by SpaCy
    """
    # For long texts, extract from first 10,000 characters only
    max_chars = 10000
    truncated_text = text[:max_chars] if len(text) > max_chars else text
    
    prompt = f"""
    Extract and identify legal entities from the following legal text. Focus on:
    
    1. PARTY: Names of parties to the agreement/case
    2. LAW: References to laws, acts, statutes, or regulations
    3. COURT: Names of courts
    4. JUDGE: Names of judges
    5. JURISDICTION: Mentioned jurisdictions
    6. LEGAL_TERM: Specialized legal terminology
    7. CASE_CITATION: Citations to legal cases
    
    For each entity, provide:
    1. The exact text of the entity
    2. The category/type from the list above
    
    Format your response as a JSON array with "text" and "type" properties for each entity.
    
    TEXT:
    {truncated_text}
    """
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
        temperature=0.2,
        response_format={"type": "json_object"}
    )
    
    result = response.choices[0].message.content
    
    # Process result to get entities with positions
    import json
    try:
        entities_data = json.loads(result)
        entities = entities_data.get("entities", [])
        
        # Add start and end positions
        processed_entities = []
        for entity in entities:
            entity_text = entity.get("text", "")
            entity_type = entity.get("type", "LEGAL_TERM")
            
            # Find all occurrences of this entity in the text
            for match in re.finditer(re.escape(entity_text), text):
                processed_entities.append({
                    "text": entity_text,
                    "start": match.start(),
                    "end": match.end(),
                    "type": entity_type
                })
                
        return processed_entities
    except json.JSONDecodeError:
        print("Error decoding JSON from OpenAI response")
        return []
    except Exception as e:
        print(f"Error processing entities: {e}")
        return []

def remove_duplicate_entities(entities):
    """
    Remove duplicate or overlapping entities, preferring more specific types
    """
    # Sort entities by start position and length (longer first for same position)
    sorted_entities = sorted(entities, key=lambda x: (x["start"], -len(x["text"])))
    
    # Define entity type priority (more specific types get higher priority)
    type_priority = {
        "CASE_CITATION": 10,
        "STATUTE": 9,
        "LAW": 8,
        "LEGAL_TERM": 7,
        "JUDGE": 6,
        "COURT": 5,
        "JURISDICTION": 4,
        "PARTY": 3,
        "PERSON": 2,
        "ORG": 1,
        "GPE": 0,
        "DATE": 0,
        "MONEY": 0,
    }
    
    # Filter out overlapping entities
    unique_entities = []
    occupied_ranges = []
    
    for entity in sorted_entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]
        priority = type_priority.get(entity_type, 0)
        
        # Check for overlaps
        overlap = False
        for i, (occ_start, occ_end, occ_priority) in enumerate(occupied_ranges):
            if not (end <= occ_start or start >= occ_end):
                # There is an overlap
                overlap = True
                # If current entity has higher priority, replace the existing one
                if priority > occ_priority:
                    # Remove the existing entity
                    occupied_ranges[i] = (start, end, priority)
                    # Find and remove the corresponding entity
                    for j, existing_entity in enumerate(unique_entities):
                        if existing_entity["start"] == occ_start and existing_entity["end"] == occ_end:
                            unique_entities[j] = entity
                            break
                break
        
        if not overlap:
            unique_entities.append(entity)
            occupied_ranges.append((start, end, priority))
    
    return unique_entities
