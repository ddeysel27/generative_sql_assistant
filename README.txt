# Generative AI SQL Assistant

A simple yet technical Generative AI project that translates **natural language questions into SQL queries**, runs them against a database, and returns the results.

## 🚀 Features
- **Text-to-SQL**: Ask questions like “Who are the top 5 customers by total purchases?”
- **LLM-powered**: Uses Generative AI (e.g., OpenAI GPT, HuggingFace models) to generate SQL.
- **Database-ready**: Works with SQLite (or PostgreSQL / MySQL with slight changes).
- **Error handling**: Detects invalid queries and provides fallback explanations.
- **Optional UI**: Streamlit dashboard for interactive querying.

---

## ⚙️ Tech Stack
- Python 3.10+
- OpenAI API (or HuggingFace Transformers)
- SQLite (default DB, easy to share)
- Pandas (for results display)
- Streamlit (optional front-end)

---

## 📂 Project Structure
generative_sql_assistant/
│── data/ # Datasets and SQLite DB
│── src/ # Core source code
│ ├── app.py # Streamlit/CLI app
│ ├── llm_utils.py # AI model + prompt handling
│ ├── sql_executor.py # Query runner
│── notebooks/ # Prototyping & experiments
│── requirements.txt # Dependencies
│── README.md # Documentation