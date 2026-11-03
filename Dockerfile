
# Use Python 3.12 as the base image
FROM python:3.12-slim

# Set the working directory inside the container
WORKDIR /app

# Prevent Python from creating .pyc files
ENV PYTHONDONTWRITEBYTECODE=1

# Display Python logs immediately
ENV PYTHONUNBUFFERED=1

# Copy the dependencies file into the container
COPY requirements.txt .

# Install all required Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application source code into the container
COPY . .

# Document the port used by Streamlit
EXPOSE 8501

# Start Streamlit and allow access from outside the container
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
