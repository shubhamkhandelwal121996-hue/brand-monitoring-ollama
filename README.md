
# 📡 BrandPulse AI
### LLM-Powered Brand Monitoring & Sentiment Analysis

BrandPulse AI is an AI-powered brand monitoring application that collects online discussions from Hacker News and uses Large Language Models (LLMs) to analyze public sentiment, identify discussion topics, detect urgent issues, and generate actionable insights.

Built with **Python, Ollama, Streamlit, and SQLite**, the application combines automated text classification with interactive analytics to help understand how brands are discussed online.

## 🚀 Features

- **Automated Mention Collection:** Fetch brand-related comments from Hacker News using the Algolia API.
- **LLM-Based Sentiment Analysis:** Classify mentions as Positive, Negative, or Neutral.
- **Topic Classification:** Identify customer complaints, product bugs, feature requests, positive reviews, competitor comparisons, and other discussions.
- **Urgency Detection:** Categorize mentions into High Urgency or Low Urgency.
- **Structured LLM Outputs:** Use Pydantic schemas to validate classification results.
- **AI-Generated Reports:** Summarize positive feedback, negative feedback, and feature requests.
- **Interactive Dashboard:** Visualize sentiment distribution and discussion topics using Plotly.
- **Persistent Storage:** Store mentions and their analysis in SQLite.
- **Concurrent Processing:** Analyze multiple mentions using Python's ThreadPoolExecutor.
- **Duplicate Prevention:** Avoid storing repeated mentions using database constraints.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Large Language Model | Ollama (`gpt-oss:120b-cloud`) |
| Frontend | Streamlit |
| Database | SQLite |
| Data Collection | Hacker News Algolia API |
| Data Processing | Pandas |
| Data Visualization | Plotly |
| Structured Output | Pydantic |
| HTTP Requests | Requests |
| Concurrency | ThreadPoolExecutor |
| Package Management | uv |

## 🏗️ System Architecture

```text
User Enters Brand Name
          |
          v
Streamlit Dashboard
          |
          v
Hacker News Algolia API
          |
          v
Fetch Brand Mentions
          |
          v
Clean & Store in SQLite
          |
          v
Pending Mentions
          |
          v
Ollama LLM Classification
          |
          +--------------------+
          |          |         |
          v          v         v
      Sentiment    Topic     Urgency
          |          |         |
          +----------+---------+
                     |
                     v
            Pydantic Validation
                     |
                     v
             SQLite Database
                     |
                     v
           Streamlit Dashboard
                     |
          +----------+----------+
          |                     |
          v                     v
   Plotly Analytics      AI-Generated Reports
```

## 📁 Project Structure

```text
brand-monitoring-ollama/
├── app.py
├── Dockerfile
├── requirements.txt
├── pyproject.toml
├── README.md
├── .gitignore
└── src/
    └── brand_monitoring_ollama/
        ├── __init__.py
        └── backend.py
```

## ⚙️ Installation

### 1. Clone the Repository

```bash
git clone https://github.com/shubhamkhandelwal121996-hue/brand-monitoring-ollama.git
cd brand-monitoring-ollama
```

### 2. Install Dependencies

Using uv:

```bash
uv sync
```

Alternatively, using pip:

```bash
pip install -r requirements.txt
```

### 3. Install Ollama

Download and install Ollama from:

https://ollama.com

Start the Ollama server if it is not already running:

```bash
ollama serve
```

### 4. Configure the LLM

The application uses the following model by default:

```text
gpt-oss:120b-cloud
```

Sign in to Ollama if required:

```bash
ollama signin
```

Test the model:

```bash
ollama run gpt-oss:120b-cloud "Hello"
```

You can configure a different Ollama model through the `OLLAMA_MODEL` environment variable.

## ▶️ Run the Application

From the project root directory:

```bash
uv run streamlit run app.py
```

Open the Streamlit dashboard at:

http://localhost:8501

## 📊 How It Works

### Step 1: Enter a Brand

Enter a brand or keyword, such as:

- Instagram
- OpenAI
- Google

### Step 2: Fetch Mentions

Select the number of Hacker News API pages to fetch.

Each page contains up to 50 results.

Fetched comments are stored in SQLite, and duplicate records are skipped.

### Step 3: Analyze Mentions

Click **Analyze Mentions** to classify unprocessed comments.

For each mention, the Ollama model predicts:

- **Sentiment:** Positive, Negative, or Neutral
- **Topic:** Discussion category
- **Urgency:** High Urgency or Low Urgency

The classification output is validated using Pydantic before being saved.

### Step 4: Explore Analytics

The dashboard displays:

- Total collected mentions
- Positive mentions
- Negative mentions
- High-urgency mentions
- Sentiment distribution
- Topic distribution
- Raw mention records

### Step 5: Generate AI Reports

Generate summaries of:

- Positive customer feedback
- Negative customer feedback
- Feature requests and suggestions

The reports help identify recurring discussion themes and potential improvement opportunities.

## 🧠 LLM Classification

The application uses a structured Pydantic model:

```python
from typing import Literal
from pydantic import BaseModel

class MentionAnalysis(BaseModel):
    sentiment: Literal["Positive", "Negative", "Neutral"]
    topic: Literal[
        "Customer Service Issue",
        "Product Defect/Bug",
        "High Price Complaint",
        "Positive Review",
        "Competitor Comparison",
        "Feature Request",
        "PR/News",
        "Other",
    ]
    urgency: Literal["High Urgency", "Low Urgency"]
```

A single LLM request produces all three classification fields, reducing the need for separate model calls.

## 🔮 Future Improvements

- Integrate additional sources such as Reddit and news platforms.
- Implement scheduled mention collection.
- Add historical sentiment trend analysis.
- Introduce automated alerts for high-urgency mentions.
- Support additional LLM providers.
- Add evaluation datasets to measure classification accuracy.

## ⚠️ Limitations

- Currently supports Hacker News as the data source.
- Classification accuracy depends on the selected LLM.
- Mentions are collected based on keyword matching and may include irrelevant discussions.
- The dashboard displays cumulative mentions stored in SQLite.
- Cloud-model inference requires an active Ollama connection and appropriate access.

