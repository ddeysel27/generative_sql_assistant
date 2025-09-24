# utils/nl2sql.py
from __future__ import annotations

# Standard
import os
import re
import difflib
import textwrap
import sqlite3
from pathlib import Path
from typing import List, Optional, Tuple
import pandas as pd
import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration

# --- tolerate both "package" and "script" imports ---
try:
    from .db_utils import get_schema_and_columns  # when imported as utils.nl2sql
except ImportError:
    from utils.db_utils import get_schema_and_columns  # fallback if run in odd contexts


# ============= Model (loaded once) =============
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_TOKENIZER: Optional[T5Tokenizer] = None
_MODEL: Optional[T5ForConditionalGeneration] = None

# allow override via env var if you want to experiment
_MODEL_ID = os.getenv("NL2SQL_MODEL", "cssupport/t5-small-awesome-text-to-sql")

# Optionally switch model at runtime before load_model()
def set_model(model_id: str) -> None:
    """Optionally switch model at runtime before load_model()."""
    global _MODEL_ID
    _MODEL_ID = model_id

#   ========== Load model (cached) ============
def load_model(force: bool = False):
    """
    Loads tokenizer + model once. Requires 'sentencepiece' installed.
    """
    global _TOKENIZER, _MODEL
    if force:
        _TOKENIZER = None
        _MODEL = None

    if _TOKENIZER is None or _MODEL is None:
        # Use a stable base tokenizer for reliability
        _TOKENIZER = T5Tokenizer.from_pretrained("t5-small")
        _MODEL = T5ForConditionalGeneration.from_pretrained(_MODEL_ID).to(_DEVICE)
        _MODEL.eval()
    return _TOKENIZER, _MODEL

# ============= Raw generation =============
def generate_sql(prompt: str, max_new_tokens: int = 256) -> str:
    tok, mdl = load_model()
    inputs = tok(prompt, return_tensors="pt", padding=True, truncation=True).to(_DEVICE)
    with torch.no_grad():
        # deterministic & concise
        out = mdl.generate(
            **inputs,
            max_length=max_new_tokens,
            do_sample=False,
            num_beams=4,
            early_stopping=True,
        )
    sql = tok.decode(out[0], skip_special_tokens=True)
    return clean_sql(sql)


# ============= Prompt + guards =============
# Common aliases to normalize in questions
ALIASES = {
    "median_income": "MedInc",
    "median house value": "MedHouseVal",
    "house age": "HouseAge",
    "avg rooms": "AveRooms",
    "average rooms": "AveRooms",
    "avg bedrooms": "AveBedrms",
    "average bedrooms": "AveBedrms",
    "lat": "Latitude",
    "lng": "Longitude",
}
# Enforce these strictly (no fuzzy matching)
STRICT_ALIASES = {
    "HorseAge": "HouseAge",
    "HypertAge": "HouseAge",
    "House_Age": "HouseAge",
    "Med_Inc": "MedInc",
    "MedianIncome": "MedInc",
    "Median_House_Value": "MedHouseVal",
    "Lat": "Latitude",
    "Long": "Longitude",
}
# Add SQL keywords and functions to avoid false positives when fixing columns 
SQL_KEYWORDS = {
    "select", "from", "where", "and", "or", "not", "in", "between", "like", "is", "null",
    "group", "by", "order", "limit", "offset", "asc", "desc", "avg", "sum", "min", "max",
    "count", "distinct", "as", "on", "join", "left", "right", "inner", "outer", "having"
}

# common SQL functions to ignore when fixing columns
SQL_FUNCS = {"avg", "sum", "min", "max", "count", "abs", "round", "upper", "lower", "coalesce"}

# ============= Normalize question =============
def normalize_question(q: str) -> str:
    out = q
    for k, v in ALIASES.items():
        out = re.sub(rf"\b{k}\b", v, out, flags=re.I)
    return out

# ============= Build prompt =============
def build_prompt(db_path: str, table: str, question: str) -> Tuple[str, List[str]]:
    ddl, cols = get_schema_and_columns(db_path, table)
    col_list = ", ".join(cols)
    question = normalize_question(question)
    rules = textwrap.dedent(f"""
    Rules:
    - Use ONLY these columns: {col_list}
    - The table name is exactly `{table}`.
    - Return ONE SQL statement, no commentary, no backticks.
    - Use sensible ranges (Latitude [-90,90], Longitude [-180,180]).
    """).strip()
    return f"tables:\n{ddl}\n{rules}\nquery for: {question}", cols

# ============= Helpers for fixing SQL =============
def tokenize_words(sql: str) -> List[str]:
    return re.findall(r"[A-Za-z_][A-Za-z_0-9]*", sql)

# ============= Enforce strict aliases =============
def apply_strict_aliases(sql: str) -> str:
    fixed = sql
    for bad, good in STRICT_ALIASES.items():
        fixed = re.sub(rf"\b{re.escape(bad)}\b", good, fixed, flags=re.I)
    return fixed

# ============= Enforce known columns =============
def enforce_columns(sql: str, cols: List[str], table_name: str) -> Tuple[str, Optional[str]]:
    col_set = {c.lower(): c for c in cols}
    table_lc = table_name.lower()
    tokens = tokenize_words(sql)
    replacements = {}

    for tok in tokens:
        tl = tok.lower()
        if tl in SQL_KEYWORDS or tl in SQL_FUNCS or tl == table_lc or tl.isdigit():
            continue
        if tl not in col_set:
            cand = difflib.get_close_matches(tl, list(col_set.keys()), n=1, cutoff=0.6)
            if cand:
                replacements[tok] = col_set[cand[0]]

    fixed = sql
    for bad, good in replacements.items():
        fixed = re.sub(rf"\b{re.escape(bad)}\b", good, fixed)

    note = None
    if replacements:
        pairs = ", ".join([f"`{b}`→`{g}`" for b, g in replacements.items()])
        note = f"🩹 Pre-fixed unknown identifiers: {pairs}"
    return fixed, note

def try_explain_sql(conn, sql: str):
    # Validate the SQL structure without fully executing it
    return pd.read_sql_query("EXPLAIN " + sql, conn)

# ============= SQL validation =============
def validate_and_fix_sql(sql: str, db_path: str, table: str, cols: list):
    sql0 = apply_strict_aliases(sql)
    sql1, pre_note = enforce_columns(sql0, cols, table)

    # --- tiny auto-repair for "FROM <column> AND ..." mistakes ---
    # e.g., "SELECT ... FROM \"Latitude\" AND Longitude > -120;"
    m = re.search(r'FROM\s+"?([A-Za-z_][\w]*)"?\s+AND\b', sql1, flags=re.I)
    if m:
        bad_token = m.group(1)
        # turn:  FROM "Latitude" AND Longitude > -120
        # into:  FROM housing WHERE Latitude AND Longitude > -120
        sql1 = re.sub(r'FROM\s+"?([A-Za-z_][\w]*)"?', f'FROM {table} WHERE {bad_token}', sql1, count=1, flags=re.I)

    conn = sqlite3.connect(db_path)
    try:
        try_explain_sql(conn, sql1)
        return sql1, pre_note
    except Exception as e:
        msg = str(e)
        m = re.search(r"no such column: ([\w_]+)", msg, re.I)
        if m:
            bad = m.group(1)
            match = difflib.get_close_matches(bad.lower(), [c.lower() for c in cols], n=1, cutoff=0.6)
            if match:
                good = next(c for c in cols if c.lower() == match[0])
                fixed = re.sub(rf"\b{re.escape(bad)}\b", good, sql1, flags=re.I)
                try:
                    try_explain_sql(conn, fixed)
                    note2 = f"🩹 Replaced `{bad}` → `{good}`"
                    note = f"{pre_note}\n{note2}" if pre_note else note2
                    return fixed, note
                except Exception:
                    pass
        return None, f"Validation failed: {msg}"
    finally:
        conn.close()



# ============= Paths / DB helper =============
BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent  # utils/ -> repo root

def get_db_path(filename: str = "housing.db") -> str:
    return str((REPO_ROOT / "data" / filename).resolve())


# ============= Cleanups & pipeline =============
def clean_sql(sql: str) -> str:
    """Strip code fences / labels; ensure single statement ending with semicolon."""
    s = sql.strip()
    s = s.replace("```sql", "").replace("```", "").strip()
    s = re.sub(r"^(SQL|Query)\s*:\s*", "", s, flags=re.I)
    if ";" in s:
        s = s.split(";")[0] + ";"
    if not s.endswith(";"):
        s += ";"
    return s

# ============= Full guarded generation =============
def generate_sql_guarded(question: str, db_path: Optional[str] = None, table: str = "housing", max_attempts: int = 2) -> str:
    db_path = db_path or get_db_path()
    prompt, cols = build_prompt(db_path, table, question)
    sql = generate_sql(prompt)

    for _ in range(max_attempts):
        fixed_sql, note = validate_and_fix_sql(sql, db_path, table, cols)
        if fixed_sql:
            if note:
                print(note)
            return clean_sql(fixed_sql)

        # regenerate with the error hint
        sql = generate_sql(
            f"{prompt}\n\nPrevious SQL:\n{sql}\n\n"
            f"Error: {note or 'Invalid SQL.'}\n"
            f"Regenerate a valid SQL using ONLY these columns: {', '.join(cols)}.\n"
            f"Return just the SQL."
        )

    return clean_sql(sql)

def build_prompt(db_path: str, table: str, question: str) -> Tuple[str, List[str]]:
    ddl, cols = get_schema_and_columns(db_path, table)
    col_list = ", ".join(cols)
    question = normalize_question(question)

    # Few-shot examples to steer the model
    few_shots = textwrap.dedent(f"""
    Examples (follow format exactly):
    Q: Average Population where Latitude > 35 AND Longitude > -120
    A: SELECT AVG(Population) FROM {table} WHERE Latitude > 35 AND Longitude > -120;

    Q: Count rows where MedInc > 5
    A: SELECT COUNT(*) FROM {table} WHERE MedInc > 5;

    Q: Average HouseAge per rounded Latitude (binning)
    A: SELECT ROUND(Latitude) AS lat_bin, AVG(HouseAge) AS avg_age
       FROM {table}
       GROUP BY lat_bin
       ORDER BY lat_bin;
    """).strip()

    rules = textwrap.dedent(f"""
    Rules:
    - Use ONLY these columns: {col_list}
    - The table name is exactly `{table}` and MUST be used after FROM.
    - Put all filters in a WHERE clause (never after FROM).
    - Return ONE SQL statement, no commentary, no backticks.
    - Use sensible ranges (Latitude [-90,90], Longitude [-180,180]).
    """).strip()

    prompt = f"tables:\n{ddl}\n{rules}\n\n{few_shots}\n\nquery for: {question}"
    return prompt, cols
