# base layer
FROM python:3.10-slim

# working dir.
WORKDIR /app

#copy requirements file
COPY requirements.txt .

# installing dependencies
RUN pip install --no-cache-dir -r requirements.txt

# copying all the files
COPY . .

# EXPOSING THE PORTS
EXPOSE 8501

# CMD TO RUN THE FILE
CMD ["streamlit","run","app.py"]
