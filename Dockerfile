# Python
FROM python:3.8-slim

# Set the working directory in the container
WORKDIR /app

# Update pip
RUN pip install --upgrade pip

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    gcc \
    libffi-dev \
    libssl-dev \
    wget \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Copy the large files from the Docker volume
COPY /data/model.pkl /app/
COPY /data/tfidf_vectorizer.pkl /app/

# Install any packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Port 80 is available to the world outside this container
EXPOSE 80

# Run app.py when the container launches
CMD ["python", "app.py"]
