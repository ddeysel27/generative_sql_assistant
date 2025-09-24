import streamlit as st
import sqlite3, pandas as pd, torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("cssupport/t5-small-awesome-text-to-sql").to(device)
model.eval()

st.title("🗄️ Text-to-SQL Assistant")

user_input = st.text_area("Ask a question in natural language:")

if st.button("Generate SQL") and user_input.strip():
    # Generate SQL
    inputs = tokenizer(user_input, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        outputs = model.generate(**inputs, max_length=512)
    sql_query = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    st.subheader("Generated SQL")
    st.code(sql_query, language="sql")
    
    # Run query against database
    conn = sqlite3.connect("../data/housing.db")
    try:
        df = pd.read_sql_query(sql_query, conn)
        st.subheader("Query Results")
        st.dataframe(df)
    except Exception as e:
        st.error(f"SQL execution failed: {e}")
    conn.close()
