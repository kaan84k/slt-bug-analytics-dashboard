"""Send email digests summarising recently discovered bug tickets.

This module reads the categorised bug data produced by the pipeline, keeps
track of which tickets have already been emailed, and sends HTML digests via
SMTP. Environment variables must provide the email credentials as documented
in the README.
"""

import os
import sys
from pathlib import Path
import pandas as pd
import smtplib

# Ensure the src directory is on the Python path when executed directly.
if __package__ is None or __package__ == "":
    project_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(project_root / "src"))

from db_utils import load_df, save_df, table_exists
from ticket_utils import assign_ticket_ids
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

# Environment variables must already be set. The script no longer loads a
# local `.env` file so it can run cleanly in GitHub Actions using secrets.

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'categorized_bugs.csv')
SENT_TABLE = 'sent_tickets'


def load_bugs(path: str) -> pd.DataFrame:
    """Load categorized bug data from ``path`` or the database."""
    if os.path.exists(path):
        df = pd.read_csv(path)
    else:
        df = load_df('categorized_bugs')
    df = assign_ticket_ids(df)
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df = df.sort_values('review_date', ascending=False)
    return df




def load_sent_tickets_db(table: str) -> set[str]:
    """Load sent ticket IDs from the SQLite database."""
    if not table_exists(table):
        return set()
    df = load_df(table)
    if 'ticket_id' not in df.columns:
        return set()
    return {str(t).strip() for t in df['ticket_id'] if str(t).strip()}


def save_sent_tickets_db(tickets: set[str], table: str) -> None:
    """Persist sent ticket IDs to the SQLite database."""
    df = pd.DataFrame({'ticket_id': sorted(tickets)})
    save_df(df, table)


def build_html_table(df: pd.DataFrame) -> str:
    """Return a styled HTML table for inclusion in the email body."""
    style = (
        "<style>"
        "table {border-collapse: collapse;width: 100%;}"\
        "th, td {border: 1px solid #ddd;padding: 8px;text-align: left;}"\
        "th {background-color: #f2f2f2;}"\
        "</style>"
    )
    table = df.to_html(index=False, escape=False)
    return f"{style}{table}"


def send_email(subject: str, html_body: str) -> None:
    """Send an HTML email using credentials from environment variables."""
    import os
    print("[DEBUG] EMAIL_USER =", os.environ.get('EMAIL_USER'))
    print("[DEBUG] EMAIL_PASS =", '*' * len(os.environ.get('EMAIL_PASS', '')))
    print("[DEBUG] EMAIL_TO =", os.environ.get('EMAIL_TO'))

    user = os.environ.get('EMAIL_USER')
    password = os.environ.get('EMAIL_PASS')
    if not user or not password:
        raise RuntimeError('EMAIL_USER and EMAIL_PASS environment variables are required')

    receiver = os.environ.get('EMAIL_TO', user)

    msg = MIMEMultipart('alternative')
    msg['Subject'] = subject
    msg['From'] = user
    msg['To'] = receiver
    msg.attach(MIMEText(html_body, 'html'))

    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(user, password)
        server.sendmail(user, receiver, msg.as_string())


def main() -> None:
    """Send an email summary of new bug tickets if any exist."""
    df = load_bugs(DATA_PATH)
    sent = load_sent_tickets_db(SENT_TABLE)

    # Treat empty table or legacy numeric IDs as first run
    first_run = len(sent) == 0
    if not first_run and all(s[4:].isdigit() for s in sent):
        print('[INFO] Resetting ticket history for new ID format.')
        sent = set()
        first_run = True

    if first_run:
        latest = df.head(10)
        if not latest.empty:
            html = build_html_table(latest[[
                'TicketID', 'bug_category', 'appVersion', 'review_date', 'review_description'
            ]])
            send_email('🚀 Initial Bug Digest: Top 10 Recent Bug Tickets', html)
        all_ids = set(df['TicketID'])
        save_sent_tickets_db(all_ids, SENT_TABLE)
        return

    new_tickets = df[~df['TicketID'].isin(sent)]
    if new_tickets.empty:
        print('No new bug tickets to send.')
        return

    html = build_html_table(new_tickets[[
        'TicketID', 'bug_category', 'appVersion', 'review_date', 'review_description'
    ]])
    subject = f"{len(new_tickets)} New Bug Tickets - SLT App"
    send_email(subject, html)

    sent.update(new_tickets['TicketID'])
    save_sent_tickets_db(sent, SENT_TABLE)


if __name__ == '__main__':
    main()
