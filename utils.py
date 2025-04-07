import re
import html

def highlight_entities_in_text(text, entities):
    """
    Highlight entities in text with different colors based on entity type
    """
    # Entity type to color mapping
    entity_colors = {
        "PERSON": "#FFC107",       # Amber
        "ORG": "#2196F3",          # Blue
        "GPE": "#4CAF50",          # Green
        "DATE": "#9C27B0",         # Purple
        "MONEY": "#F44336",        # Red
        "LAW": "#FF9800",          # Orange
        "COURT": "#795548",        # Brown
        "JUDGE": "#607D8B",        # Blue Gray
        "STATUTE": "#E91E63",      # Pink
        "REGULATION": "#009688",   # Teal
        "CASE_CITATION": "#673AB7", # Deep Purple
        "LEGAL_TERM": "#3F51B5",   # Indigo
        "PARTY": "#8BC34A",        # Light Green
        "JURISDICTION": "#00BCD4"  # Cyan
    }
    
    # Default color for unknown entity types
    default_color = "#9E9E9E"  # Gray
    
    # Sort entities by start position in reverse order
    # This ensures that we process entities from the end of the text first
    # to avoid issues with changing text positions
    sorted_entities = sorted(entities, key=lambda x: x["start"], reverse=True)
    
    # Convert text to HTML-safe
    html_text = html.escape(text)
    
    # Insert highlighting spans
    for entity in sorted_entities:
        start = entity["start"]
        end = entity["end"]
        entity_type = entity["type"]
        entity_text = html.escape(text[start:end])
        
        # Get color for entity type
        color = entity_colors.get(entity_type, default_color)
        
        # Create highlighted span
        highlighted_span = f'<span style="background-color: {color}; padding: 2px; border-radius: 3px;" title="{entity_type}">{entity_text}</span>'
        
        # Insert span into text
        html_text = html_text[:start] + highlighted_span + html_text[end:]
    
    # Convert newlines to <br> tags
    html_text = html_text.replace('\n', '<br>')
    
    # Wrap in div for styling
    return f'<div style="font-family: monospace; white-space: pre-wrap; line-height: 1.5;">{html_text}</div>'

def truncate_text(text, max_length=1000):
    """
    Truncate text to maximum length while preserving whole words
    """
    if len(text) <= max_length:
        return text
    
    # Find the last space within the max length
    last_space = text[:max_length].rfind(' ')
    if last_space == -1:
        # If no space found, just cut at max_length
        return text[:max_length] + "..."
    
    return text[:last_space] + "..."

def format_json_for_display(json_data):
    """
    Format JSON data for display in Streamlit
    """
    if not json_data:
        return ""
    
    formatted_text = ""
    for key, value in json_data.items():
        formatted_text += f"**{key}**: "
        if isinstance(value, dict):
            formatted_text += "\n"
            for sub_key, sub_value in value.items():
                formatted_text += f"  - {sub_key}: {sub_value}\n"
        else:
            formatted_text += f"{value}\n"
    
    return formatted_text
