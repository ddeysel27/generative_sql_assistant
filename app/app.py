# app/app.py
import os, sys

# Resolve repo root so paths work no matter where you launch Streamlit
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DB_PATH = os.path.join(REPO_ROOT, "data", "housing.db")
TABLE = "housing"
# Ensure 'utils' is importable when running directly
if REPO_ROOT not in sys.path:
    sys.path.append(REPO_ROOT)

# Streamlit app
import streamlit as st, pandas as pd
# Our utils
from utils.nl2sql import load_model, generate_sql_guarded
from utils.db_utils import get_schema_and_columns, run_sql

# Warm up the model at startup (cached so only runs once)
@st.cache_resource(show_spinner=True)
def _warm_model():
    load_model()
    return True
_ = _warm_model()

# Streamlit UI
st.set_page_config(page_title="Text-to-SQL Assistant", page_icon="🗄️", layout="wide")
st.title("🗄️ Text-to-SQL Assistant")

# Sidebar: show schema and the Existing Columns
with st.sidebar:
    st.header("Schema")
    try:
        ddl, cols = get_schema_and_columns(DB_PATH, TABLE)
        st.code(ddl, language="sql")
        st.markdown("**Columns**")
        st.write(cols)
    except Exception as e:
        st.error(f"Could not read schema: {e}")

# Main area: input question, generate SQL, run & show results
st.markdown("Ask a question about the **housing** database:")
default_q = "Average house age for blocks with MedInc > 5 and Latitude > 35"
user_q = st.text_area("Natural language question", value=default_q, height=100)

# Buttons: Generate SQL & Run / Just show SQL
col1, col2 = st.columns([1,1])
with col1:
    run_btn = st.button("Generate SQL & Run", type="primary")
with col2:
    show_sql_only = st.checkbox("Just show SQL (don’t execute)")

# If button pressed and question non-empty: generate SQL
if run_btn and user_q.strip():
    with st.spinner("Generating SQL..."):
        sql = generate_sql_guarded(user_q, db_path=DB_PATH, table=TABLE)

    st.subheader("Generated SQL")
    st.code(sql, language="sql")

    if not show_sql_only:
        try:
            df = run_sql(DB_PATH, sql)
            st.subheader("Query Results")
            if df.empty:
                st.info("Query returned 0 rows.")
            else:
                st.dataframe(df, use_container_width=True)
        except Exception as e:
            st.error(f"SQL execution failed: {e}")

