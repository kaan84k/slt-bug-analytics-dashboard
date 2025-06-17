import pandas as pd
from pathlib import Path
from datetime import datetime

from src.db_utils import load_df, save_df, table_exists

ATTENDED_PATH = Path(__file__).resolve().parents[2] / 'data' / 'attended_tickets.csv'
ATTENDED_TABLE = 'attended_tickets'


def load_attended() -> pd.DataFrame:
    """Return attended tickets as DataFrame with TicketID and attended_at."""
    if ATTENDED_PATH.exists():
        return pd.read_csv(ATTENDED_PATH)
    if table_exists(ATTENDED_TABLE):
        return load_df(ATTENDED_TABLE)
    return pd.DataFrame(columns=["TicketID", "attended_at"])


def save_attended(df: pd.DataFrame) -> None:
    ATTENDED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(ATTENDED_PATH, index=False)
    save_df(df, ATTENDED_TABLE)


def mark_tickets_attended(ticket_ids: list[str]) -> None:
    """Add ticket IDs to the attended list if not already present."""
    if not ticket_ids:
        return
    df = load_attended()
    existing = set(df['TicketID'])
    new_rows = [
        {'TicketID': tid, 'attended_at': datetime.utcnow().isoformat()}
        for tid in ticket_ids if tid not in existing
    ]
    if new_rows:
        df = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
        save_attended(df)
