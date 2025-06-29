import hashlib
import pandas as pd

def assign_ticket_ids(df: pd.DataFrame) -> pd.DataFrame:
    """Return DataFrame with stable TicketID column."""
    def _make_id(row: pd.Series) -> str:
        base = f"{row.get('review_description','')}_{row.get('review_date','')}_{row.get('appVersion','')}"
        digest = hashlib.sha1(base.encode('utf-8')).hexdigest()[:8].upper()
        return f"BUG-{digest}"
    df = df.copy()
    df['TicketID'] = df.apply(_make_id, axis=1)
    return df
