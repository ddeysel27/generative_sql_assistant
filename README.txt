#  Generative AI Text-to-SQL Assistant

A lightweight but highly practical Generative AI tool that converts **natural-language questions into SQL queries**, executes them on a database, and returns clean, readable results.  
Designed for demos, data-analysis workflows, and deploying an interactive **Streamlit UI**.

---

##  Key Features

###  Text → SQL Generation  
Ask natural questions like:  
**“List the total revenue per customer for 2023.”**  
Get a valid SQL query instantly.

###  LLM-Powered Query Builder  
Supports:
- OpenAI GPT models  
- Local / HuggingFace LLMs  

###  Database Ready  
- **SQLite** by default (portable & shareable)  
- Optional: PostgreSQL or MySQL with minor config changes

###  Smart Error Handling  
- Detects malformed SQL  
- Retries with refined prompts  
- Explains errors clearly  

###  Optional Streamlit Web UI  
Run the generator in a simple browser app.

###  Clean Results Output  
Delivered as:
- Pandas DataFrames (CLI)
- Streamlit Tables (UI)

---

##  Tech Stack

- Python **3.10+**
- **OpenAI API** *(or HuggingFace Transformers)*
- **SQLite** (default)
- **Pandas**
- **SQLAlchemy**
- **Streamlit** *(optional front-end)*

---

## ⚙️ How It Works

### 1. User Inputs a Natural Language Question  
Example:  
> “Show the top 10 orders by total revenue.”

### 2. LLM Generates SQL  
Model receives:
- Database schema  
- Sample queries  
- SQL formatting rules  
- Dialect instructions (SQLite default)

### 3️. SQL is Executed Safely  
Handled by `sql_executor.py`:
- Input sanitization  
- SQLAlchemy execution  
- Error catching  
- Fallback explanations  

### 4️. Results Are Returned  
Displayed as:
- **Pandas DataFrame** (CLI)
- **Streamlit table** (UI)

---

##  Running the Project

### **1. Install dependencies**
```bash
pip install -r requirements.txt


2. Add your API key

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
