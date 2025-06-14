import os
import pandas as pd
import smtplib
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover - python-dotenv optional
    def load_dotenv(*_args, **_kwargs):
        pass

# Load environment variables if a .env file is present
load_dotenv()

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'categorized_bugs.csv')
SENT_FILE = os.path.join(os.path.dirname(__file__), 'sent_tickets.txt')


def load_bugs(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.reset_index(drop=True)
    df['TicketID'] = [f"BUG-{i+1001:04d}" for i in range(len(df))]
    df['review_date'] = pd.to_datetime(df['review_date'], errors='coerce')
    df = df.sort_values('review_date', ascending=False)
    return df


def load_sent_tickets(path: str) -> set[str]:
    if not os.path.exists(path):
        return set()
    with open(path, 'r') as f:
        return {line.strip() for line in f if line.strip()}


def save_sent_tickets(tickets: set[str], path: str) -> None:
    with open(path, 'w') as f:
        for ticket in sorted(tickets):
            f.write(f"{ticket}\n")


def build_html_table(df: pd.DataFrame) -> str:
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
    df = load_bugs(DATA_PATH)
    sent = load_sent_tickets(SENT_FILE)

    if not os.path.exists(SENT_FILE):
        latest = df.head(10)
        if not latest.empty:
            html = build_html_table(latest[[
                'TicketID', 'bug_category', 'appVersion', 'review_date', 'review_description'
            ]])
            send_email('🚀 Initial Bug Digest: Top 10 Recent Bug Tickets', html)
        save_sent_tickets(set(df['TicketID']), SENT_FILE)
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
    save_sent_tickets(sent, SENT_FILE)


if __name__ == '__main__':
    main()
