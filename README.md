# NLP_Enron_Email_Sentiment_NER_Analysis_Flask_App
This repository contains the EnronEmailInsights application, a Flask-based web application designed for real-time sentiment analysis and entity detection within emails related to the oil and gas sector, specifically focusing on Enron's business activities.
====================================================
Email Analysis Flask Application - README
====================================================

Author: Shadi Momtahen
Date: July 12, 2024

====================================================
Overview
====================================================

This project implements a Flask web application for email sentiment analysis and entity detection related to Enron's Oil & Gas business. The application provides a user-friendly interface where users can input email content and receive JSON-formatted results containing sentiment analysis, a flag indicating relevance to Enron's Oil & Gas, and detected persons or organizations.

====================================================
Setup Instructions
====================================================

1. Requirements:
   - Python 3.x
   - Flask
   - NumPy
   - Pandas
   - Scikit-learn
   - TensorFlow (for LSTM model)
   - PyTorch (for neural networks)
   - Spark (optional, for PySpark integration)

2. Installation:
   a. Clone the repository:
      ```
      git clone <repository_url>
      cd email_analysis_flask_app
      ```

   b. Install dependencies:
      ```
      pip install -r requirements.txt
      ```

3. Running the Application:
   a. Set up the Flask application:
      ```
      python app.py
      ```

   b. Access the application:
      Open a web browser and go to http://localhost:8080/

====================================================
Usage
====================================================

1. Home Page:
   - Navigate to http://localhost:8080/ to access the home page.
   - Enter email content into the form on the page.

2. Submitting Email Content:
   - Fill out the form with email text and submit.

3. Result Page:
   - View the sentiment analysis (Positive/Negative).
   - Check if the email is related to Enron's Oil & Gas.
   - See a list of detected persons or organizations related to the oil and gas sector.

4. JSON Response:
   - Results are returned in JSON format:
     ```
     {
       "sentiment": "Positive",
       "is_related_to_oil_gas": true,
       "entities": {
         "Persons": ["John Doe", "Jane Smith"],
         "Organizations": ["Enron Corporation"]
       }
     }
     ```

====================================================
Additional Notes
====================================================

- The application uses machine learning models for sentiment analysis and entity detection.
- Please ensure all dependencies are installed and compatible with your Python environment.
- For production deployment, Dockerizing the application allows for containerized execution, which can be beneficial for scalability and environment consistency.
- If deploying on a system with GPU capabilities, ensure Docker configuration includes GPU support for enhanced performance with deep learning models.


