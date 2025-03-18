FROM python:3.9-slim

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install the correct scikit-learn version that matches your model
RUN pip install --no-cache-dir scikit-learn==1.5.1

# Create models directory if it doesn't exist
RUN mkdir -p /app/models

# Copy application code
COPY . .

# List the contents of the app and models directories for debugging
RUN echo "App directory contents:" && ls -la . && \
    echo "Models directory contents:" && ls -la models || echo "Models directory not found or empty"

# Expose the port
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app
ENV MODEL_PATH=models/stroke_prediction_model_20250318_122920.joblib
ENV PYTHONUNBUFFERED=1

# Run the application
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000", "--log-level", "debug"]