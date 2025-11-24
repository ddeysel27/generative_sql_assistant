Generative AI Text-to-SQL Assistant

A lightweight but highly practical Generative AI tool that converts natural-language questions into SQL queries, executes them on a database, and returns clean, readable results. Designed for demos, data-analysis workflows, and deploying an interactive Streamlit UI.

Key Features

Text → SQL Generation
Ask: “List the total revenue per customer for 2023” — get a valid SQL query instantly.

LLM-Powered Query Builder
Supports OpenAI GPT models or local/HuggingFace models.

Database Ready
Default: SQLite for portability
Optional: PostgreSQL or MySQL with minor config changes.

Smart Error Handling
Detects malformed SQL, retries with improved prompts, and explains errors.

Optional Streamlit Web UI
Run the whole system in a simple browser app.

Clean Results Output
Results returned as Pandas DataFrames or Streamlit tables.

Tech Stack

Python 3.10+

OpenAI API (or HuggingFace Transformers for local models)

SQLite (default)

Pandas

SQLAlchemy

Streamlit (optional UI layer)

⚙️ How It Works
1. User Inputs a Natural Language Question

Example:

“Show the top 10 orders by total revenue.”

2. LLM Generates SQL

The model is prompted with:

database schema

sample queries

formatting rules

SQL dialect (SQLite by default)

3. SQL is Executed Safely

sql_executor.py handles:

input sanitation

SQLAlchemy execution

error catching

fallback explanations

4. Results are Returned

Output is displayed as:

Pandas DataFrame (CLI)

Streamlit Table (UI)

▶️ Running the Project
1. Install dependencies
pip install -r requirements.txt

2. Add your OpenAI (or HF) API key

Create a .env file:

OPENAI_API_KEY=your_key_here

3. Start the Streamlit app
streamlit run src/app.py

4. Or run in CLI mode
python src/app.py --cli

🧩 Example Query

Input:

"How many unique products were sold last year?"

Generated SQL:

SELECT COUNT(DISTINCT product_id)
FROM orders
WHERE order_date >= '2023-01-01';


Output:
A clean Pandas DataFrame.
