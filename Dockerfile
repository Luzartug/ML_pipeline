
# Use an official Python runtime as a parent image
FROM python:3.11

# Set the working directory in the container
WORKDIR /app

# Copy only the necessary files and directories
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 8501 available to the world outside this container
# Streamlit uses 8501 by default, adjust if needed
EXPOSE 8501

# Make port 5000 available for MLflow UI, adjust if needed
EXPOSE 3000

# Define environment variable for MLflow
ENV MLFLOW_TRACKING_URI=/app/mlruns

# Command to run the MLflow UI in the background and start the Streamlit app
CMD mlflow ui --backend-store-uri /app/mlruns --host 0.0.0.0 --port 3000 & streamlit run app.py --server.port=8501
