def extract_entities(text):
    logging.debug(f"Extracting entities from text: {text}")
    # Use Spacy for entity extraction
    doc = nlp(text)
    persons = [ent.text for ent in doc.ents if ent.label_ == 'PERSON']
    organizations = [ent.text for ent in doc.ents if ent.label_ == 'ORG']
    logging.debug(f"Extracted persons: {persons}")
    logging.debug(f"Extracted organizations: {organizations}")
    return {
        'persons': list(set(persons)),
        'organizations': list(set(organizations))
    }
