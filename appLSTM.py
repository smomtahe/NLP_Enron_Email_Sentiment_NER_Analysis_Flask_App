import os
import re
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from flask import Flask, request, jsonify, render_template
import spacy

# Initialize Flask and spaCy
app = Flask(__name__)
nlp = spacy.load("en_core_web_sm")

# Define LSTM model configuration
vocab_size = 20000  # Size of vocabulary
maxlen = 200  # Maximum length of text sequences
embedding_dim = 128  # Embedding dimensions

# Initialize Tokenizer
tokenizer = Tokenizer(num_words=vocab_size)
# LSTM model initialization
model_lstm = Sequential([
    Embedding(vocab_size, embedding_dim, input_length=maxlen),
    LSTM(64),
    Dense(1, activation='sigmoid')
])
model_lstm.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Process email file to extract content
def process_email_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as file:
            content = file.read()
            # Regex to extract the email body (simplified for demonstration)
            body = re.search(r"X-FileName:.*?[\r\n]+(.*)", content, re.S)
            if body:
                email_text = body.group(1)
                return email_text
            return ""
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return ""

# Endpoint to process email and predict sentiment and extract entities
@app.route('/predict', methods=['POST'])
def predict_email_sentiment():
    file_path = request.form['file_path']
    email_text = process_email_file(file_path)
    if email_text:
        # Tokenize and pad the text
        seq = tokenizer.texts_to_sequences([email_text])
        padded_seq = pad_sequences(seq, maxlen=maxlen)
        # Predict using the LSTM model
        prediction = model_lstm.predict(padded_seq)[0]
        sentiment = 'Positive' if prediction > 0.5 else 'Negative'
        
        # Perform NER using spaCy
        doc = nlp(email_text)
        entities = [(ent.text, ent.label_) for ent in doc.ents]

        return jsonify({
            'sentiment': sentiment,
            'entities': entities
        })
    return jsonify({'error': 'Unable to process the file'})

# Main block to run the app
if __name__ == '__main__':
    app.run(debug=True)
