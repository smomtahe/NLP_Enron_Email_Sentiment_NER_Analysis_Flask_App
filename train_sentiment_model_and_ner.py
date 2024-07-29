import pandas as pd
import spacy
import string
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

def train_sentiment_model_and_ner(data):
    # Preprocessing function
    def preprocess_text(text):
        # Remove punctuation
        text = text.translate(str.maketrans('', '', string.punctuation))
        # Remove numbers
        text = re.sub(r'\d+', '', text)
        # Lowercase
        text = text.lower()
        return text
    
    # Apply preprocessing
    df = pd.DataFrame(data)
    df['text'] = df['text'].apply(preprocess_text)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)
    
    # Vectorize the text
    vectorizer = TfidfVectorizer()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train the random forest model
    model = RandomForestClassifier()
    model.fit(X_train_tfidf, y_train)
    
    # Predict on the test set
    y_pred = model.predict(X_test_tfidf)
    
    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['Negative', 'Positive'], zero_division=0)
    
    print(f"Model accuracy: {accuracy}")
    print("Classification Report:")
    print(report)
    
    # Save the model and vectorizer
    joblib.dump(model, 'random_forest_model.joblib')
    joblib.dump(vectorizer, 'tfidf_vectorizer.joblib')
    
    # Test texts for Named Entity Recognition
    test_texts = [
        "John Doe is a software engineer at Google.",
        "Microsoft Corporation is one of the leading tech companies.",
        "This is a test email to analyze the sentiment and extract entities. It mentions oil and gas companies and involves energy discussions."
    ]
    
    # Load spaCy model
    nlp = spacy.load('en_core_web_sm')
    
    for text in test_texts:
        doc = nlp(text)
        persons = [ent.text for ent in doc.ents if ent.label_ == "PERSON"]
        organizations = [ent.text for ent in doc.ents if ent.label_ == "ORG"]
        print(f"Text: {text}\nPersons: {persons}\nOrganizations: {organizations}\n")

# Example dataset
data = {
    'text': [
        "I love this product, it is amazing!",
        "I hate this service, it is terrible!",
        "This is a neutral statement.",
        "The transaction is successful and we are very happy.",
        "The meeting was awful and everything went wrong.",
        "I enjoyed the experience, it was fantastic!",
        "This is the worst product I have ever bought.",
        "Absolutely delighted with the outcome!",
        "I am very unhappy with the service provided.",
        "The service was satisfactory.",
        "This is a fantastic development!",
        "I'm disappointed with the results.",
        "The new update is great and works well.",
        "I had a horrible experience with the product.",
        "Customer support was very helpful and friendly.",
        "I'm never buying this again, terrible quality.",
        "The quality exceeded my expectations.",
        "I'm not satisfied with this purchase.",
        "The team did an excellent job!",
        "Very poor customer service experience.",
        "I love the new features!",
        "This is not what I expected, I'm dissatisfied.",
        "Fantastic customer service, very pleased!",
        "I can't believe how bad this is.",
        "The performance is exceptional, well done!",
        "I'm frustrated with the constant issues.",
        "Wonderful product, will recommend to others.",
        "This is unacceptable, very poor quality.",
        "Exceeded all my expectations, great job!",
        "I want a refund, very disappointed.",
        "Very happy with the results, thank you!",
        # Adding more positive and negative samples
        "The product quality is top-notch.",
        "Terrible experience with this service.",
        "I'm so happy with my purchase!",
        "Worst decision I've ever made.",
        "I can't express how satisfied I am.",
        "I'm utterly disappointed.",
        "Great value for money!",
        "It was a complete waste of time.",
        "Highly recommend this to everyone!",
        "Avoid at all costs.",
        "Exceptional quality and fast shipping.",
        "The product broke after one use.",
        "Fantastic experience, will buy again.",
        "Very poor craftsmanship, not recommended.",
        "I'm thrilled with my purchase!",
        "The service was abysmal and slow."
    ],
    'label': [
        1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 1,
        0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1,
        1, 0, 1, 0, 1, 0, 1, 0, 1, 0,
        1, 0, 1, 0, 1, 0
    ]  # 1 for positive, 0 for negative
}

# Train the sentiment analysis model and perform NER
train_sentiment_model_and_ner(data)
