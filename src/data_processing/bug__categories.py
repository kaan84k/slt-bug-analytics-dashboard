bug_keywords = {
    "Login Error": ["login", "log in", "sign in", "cannot access", "authentication", "credentials"],
    "UI Issue": ["layout", "screen", "button", "display", "alignment", "responsive", "interface"],
    "Crash/Freeze": ["crash", "hang", "freeze", "stuck", "unresponsive", "not responding"],
    "Payment Issue": ["payment", "bill", "topup", "recharge", "transaction", "fail", "credit card"],
    "Slow Performance": ["slow", "lag", "delay", "loading", "takes time", "wait", "performance"],
    "Notification Problem": ["notification", "alert", "reminder", "not getting", "missing notifications"],
    "Update Issue": ["update", "updated", "version", "after update", "since update", "new version"],
    "Server Error": ["server error", "server down", "server not responding", "cannot connect", "connection error", "500", "503"],
    "Network/Connection": ["network", "connection", "wifi", "internet", "signal", "offline", "no service", "poor connection", "weak signal", "disconnected"],
    "Other": []
}

import pandas as pd
import re

# Load prioritized bug data
df = pd.read_csv("data/prioritized_bugs.csv")

# Preprocess review text
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['cleaned_description'] = df['review_description'].apply(preprocess)

def categorize_review(text):
    text = text.lower()
    for category, keywords in bug_keywords.items():
        if any(keyword.lower() in text for keyword in keywords):
            return category
    return "Other"

df['bug_category'] = df['cleaned_description'].apply(categorize_review)

df[['review_description', 'bug_category', 'review_date', 'appVersion']].to_csv("data/categorized_bugs.csv", index=False)
