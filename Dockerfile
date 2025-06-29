# Use an official Python runtime as a parent image
FROM python:3.11-slim

# Create app directory
WORKDIR /app

# Install dependencies
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
    && pip install --no-cache-dir streamlit

# Copy application code
COPY . .

# Streamlit settings for headless operation
ENV STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_ENABLE_CORS=false

EXPOSE 8501

# Default command starts the Streamlit dashboard
CMD ["streamlit", "run", "src/dashboard/app.py"]
