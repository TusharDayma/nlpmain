"""
Named Entity Recognition (NER) System
======================================
This script demonstrates NER using Spacy to extract:
- Person Names
- Locations
- Organizations
- Dates
- Monetary Values

Author: NLP Practicals
Date: 2026
"""

import spacy
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')


def load_spacy_model():
    """Load Spacy English model for NER"""
    try:
        nlp = spacy.load("en_core_web_sm")
        print("✓ Spacy model loaded successfully\n")
        return nlp
    except OSError:
        print("Spacy model not found. Installing 'en_core_web_sm'...")
        import os
        os.system("python -m spacy download en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        return nlp


def extract_entities(nlp, text):
    """
    Extract named entities from text using Spacy
    
    Args:
        nlp: Spacy NLP model
        text: Input text for NER
        
    Returns:
        Dictionary containing categorized entities
    """
    doc = nlp(text)
    
    entities_dict = defaultdict(list)
    
    # Map Spacy entity labels to our categories
    entity_mapping = {
        'PERSON': 'Person Names',
        'ORG': 'Organizations',
        'GPE': 'Locations',  # Geopolitical entities
        'LOC': 'Locations',  # Locations
        'DATE': 'Dates',
        'TIME': 'Times',
        'MONEY': 'Monetary Values',
        'QUANTITY': 'Quantities',
        'PERCENT': 'Percentages',
        'PRODUCT': 'Products',
        'EVENT': 'Events',
        'LAW': 'Laws'
    }
    
    # Extract entities
    for ent in doc.ents:
        category = entity_mapping.get(ent.label_, ent.label_)
        entities_dict[category].append({
            'text': ent.text,
            'label': ent.label_,
            'start': ent.start_char,
            'end': ent.end_char
        })
    
    return entities_dict


def display_results(text, entities_dict, title="NER Results"):
    """
    Display NER results in a formatted manner
    
    Args:
        text: Original text
        entities_dict: Dictionary of extracted entities
        title: Title for the results section
    """
    print("=" * 80)
    print(f"  {title}")
    print("=" * 80)
    print(f"\nOriginal Text:\n{text}\n")
    print("-" * 80)
    print("EXTRACTED ENTITIES:\n")
    
    # Priority order for display
    priority_order = [
        'Person Names',
        'Organizations',
        'Locations',
        'Dates',
        'Times',
        'Monetary Values',
        'Quantities',
        'Percentages',
        'Products',
        'Events',
        'Laws'
    ]
    
    found_entities = False
    for category in priority_order:
        if category in entities_dict and entities_dict[category]:
            found_entities = True
            print(f"📍 {category}:")
            unique_entities = set()
            for entity in entities_dict[category]:
                unique_entities.add(entity['text'])
            for entity_text in sorted(unique_entities):
                print(f"   • {entity_text}")
            print()
    
    if not found_entities:
        print("No entities found in the text.")
    
    print("=" * 80)
    print()


def sample_texts():
    """Return sample texts for NER demonstration"""
    return {
        "technology_article": """
        Apple Inc. was founded by Steve Jobs, Steve Wozniak, and Ronald Wayne on April 1, 1976
        in Los Altos, California. The company revolutionized technology with the iPhone, which was
        first released on June 29, 2007. Tim Cook became the CEO in August 2011 and led Apple to
        a market capitalization exceeding $2.5 trillion. The headquarters is located at Apple Park
        in Cupertino. Apple announced a partnership with Microsoft on December 10, 2020, worth
        approximately $1 billion. The company operates in over 150 countries globally.
        """,
        
        "finance_article": """
        Elon Musk announced that Tesla's quarterly revenue reached $25 billion in Q3 2024. The
        stock price increased by 15.5% after the announcement on December 15, 2024. JPMorgan Chase
        upgraded Tesla's rating to "Buy" with a price target of $350 per share. The headquarters
        in Austin, Texas saw a 20% increase in production. Goldman Sachs predicted that the EV
        market could reach $500 billion by 2030. Tesla's expansion into Berlin and Shanghai has
        been successful since January 2023.
        """,
        
        "historical_article": """
        Winston Churchill was born on November 30, 1874 in Oxfordshire, England. He served as Prime
        Minister of the United Kingdom from May 1940 to July 1945 during World War II. Churchill
        met with Franklin D. Roosevelt, the President of the United States, in Cairo on November 28, 1943.
        He won the Nobel Prize in Literature on December 10, 1953. His legacy shaped modern British
        politics, and he is buried in the churchyard of St Martin's Church in Bladon, Oxfordshire.
        """
    }


def main():
    """Main execution function"""
    print("\n" + "=" * 80)
    print("  NAMED ENTITY RECOGNITION (NER) SYSTEM")
    print("=" * 80 + "\n")
    
    # Load Spacy model
    nlp = load_spacy_model()
    
    # Get sample texts
    samples = sample_texts()
    
    # Process each sample
    print("Processing sample texts...\n")
    for sample_name, text in samples.items():
        # Clean up text
        text = ' '.join(text.split())
        
        # Extract entities
        entities = extract_entities(nlp, text)
        
        # Display results
        title = f"NER Results - {sample_name.replace('_', ' ').title()}"
        display_results(text, entities, title)
    
    # Interactive mode
    print("\n" + "=" * 80)
    print("  INTERACTIVE MODE")
    print("=" * 80)
    print("\nEnter your own text for NER analysis (type 'quit' to exit):\n")
    
    while True:
        user_input = input("Enter text: ").strip()
        
        if user_input.lower() == 'quit':
            print("\nThank you for using the NER System!")
            break
        
        if not user_input:
            print("Please enter some text.\n")
            continue
        
        # Extract and display entities for user input
        entities = extract_entities(nlp, user_input)
        display_results(user_input, entities, "Your Text - NER Results")


if __name__ == "__main__":
    main()
