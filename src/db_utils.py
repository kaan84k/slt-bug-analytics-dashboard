import sqlite3
from pathlib import Path
import pandas as pd

# Default database location inside the repository's data directory
DB_PATH = Path(__file__).resolve().parents[1] / 'data' / 'bug_data.db'


def get_connection(db_path: Path = DB_PATH) -> sqlite3.Connection:
    """Return a connection to the SQLite database."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    return sqlite3.connect(db_path)


def save_df(df: pd.DataFrame, table_name: str, conn: sqlite3.Connection | None = None) -> None:
    """Save a DataFrame to a SQLite table, replacing existing data."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    df.to_sql(table_name, conn, if_exists='replace', index=False)
    if close:
        conn.close()


def load_df(table_name: str, conn: sqlite3.Connection | None = None) -> pd.DataFrame:
    """Load a DataFrame from a SQLite table."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    df = pd.read_sql(f'SELECT * FROM {table_name}', conn)
    if close:
        conn.close()
    return df


def table_exists(table_name: str, conn: sqlite3.Connection | None = None) -> bool:
    """Check if a table exists in the database."""
    close = False
    if conn is None:
        conn = get_connection()
        close = True
    cur = conn.cursor()
    cur.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
    exists = cur.fetchone() is not None
    if close:
        conn.close()
    return exists
