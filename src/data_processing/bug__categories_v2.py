bug_keywords = {
    "Login Error": ["login", "log in", "sign in", "cannot access", "authentication", "credentials", "password", "username", "account", "invalid", "oops something went wrong", "verification", "10 digits", "can't log in"],
    "UI Issue": ["layout", "screen", "button", "display", "alignment", "responsive", "interface"],
    "Crash/Freeze": ["crash", "hang", "freeze", "stuck", "unresponsive", "not responding"],
    "Payment Issue": ["payment", "bill", "topup", "recharge", "transaction", "fail", "credit card"],
    "Slow Performance": ["slow", "lag", "delay", "loading", "takes time", "wait", "performance"],
    "Notification Problem": ["notification", "alert", "reminder", "not getting", "missing notifications"],
    "Update Issue": ["update", "updated", "version", "after update", "since update", "new version"],
    "Server Error": ["server error", "server down", "server not responding", "cannot connect", "connection error", "500", "503"],
    "Contact Change Issue": ["contact", "phone number", "mobile number", "change number", "update contact", "contact information", "contact details", "phone details", "cannot change", "number change"],
    "Other": []
}

import pandas as pd
import re

# Load prioritized bug data
df = pd.read_csv("data/reclassified_bugs_with_sbert.csv")

# Preprocess review text
def preprocess(text):
    text = str(text).lower()
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['cleaned_description'] = df['review_description'].apply(preprocess)

def categorize_review(text):
    for category, keywords in bug_keywords.items():
        if any(kw in text for kw in keywords):
            return category
    return "Other"

df['bug_category'] = df['cleaned_description'].apply(categorize_review)

df[['review_description', 'bug_category', 'review_date', 'appVersion']].to_csv("data/categorized_bugs.csv", index=False)
