# tests/test_email_analysis.py

import pytest
from app import parse_email, clean_text, analyze_sentiment, extract_entities, is_related_to_oil_gas

# Sample raw email content for testing
RAW_EMAIL = """
From: sender@example.com
To: recipient@example.com
Subject: Test Email
This is a test email to analyze the sentiment and extract entities. It mentions oil and gas companies and involves energy discussions.
"""

def test_parse_email():
    email_body = parse_email(RAW_EMAIL)
    assert isinstance(email_body, str)
    assert len(email_body) > 0

def test_clean_text():
    email_body = parse_email(RAW_EMAIL)
    cleaned_text = clean_text(email_body)
    assert "\\n" not in cleaned_text  # Ensure newlines are removed
    assert "X-From:" not in cleaned_text  # Ensure specific headers are removed

def test_analyze_sentiment():
    email_body = parse_email(RAW_EMAIL)
    cleaned_text = clean_text(email_body)
    sentiment = analyze_sentiment(cleaned_text)
    assert sentiment in ['POSITIVE', 'NEGATIVE']

def test_extract_entities():
    email_body = parse_email(RAW_EMAIL)
    cleaned_text = clean_text(email_body)
    entities = extract_entities(cleaned_text)
    assert isinstance(entities, dict)
    assert 'persons' in entities
    assert 'organizations' in entities

def test_is_related_to_oil_gas():
    email_body = parse_email(RAW_EMAIL)
    cleaned_text = clean_text(email_body)
    related_to_oil_gas = is_related_to_oil_gas(cleaned_text)
    assert isinstance(related_to_oil_gas, bool)

# If running this file directly, execute pytest command
if __name__ == "__main__":
    pytest.main()
