bug_keywords = {
    "Login Error": ["login", "log in", "sign in", "cannot access", "authentication", "credentials", "password", "username", "account", "invalid", "oops something went wrong", "verification", "10 digits", "can't log in"],
    "UI Issue": ["layout", "screen", "button", "display", "alignment", "responsive", "interface"],
    "Crash/Freeze": ["crash", "hang", "freeze", "stuck", "unresponsive", "not responding"],
    "Payment Issue": ["payment", "bill", "topup", "recharge", "transaction", "fail", "credit card"],
    "Slow Performance": ["slow", "lag", "delay", "loading", "takes time", "wait", "performance"],
    "Notification Problem": ["notification", "alert", "reminder", "not getting", "missing notifications"],
    "Update Issue": ["update", "updated", "version", "after update", "since update", "new version"],
    "Server Error": ["server error", "server down", "server not responding", "cannot connect", "connection error", "500", "503"],
    "Other": []  # Network/Connection can be merged or kept separate if distinct enough
    # For this consolidation, Network/Connection keywords are removed as they are not in bug_keywords_v2.
    # If they were important, they'd be merged into "Other" or a new relevant category.
}

import pandas as pd
import re

# Load prioritized bug data
df = pd.read_csv("prioritized_bugs.csv")

# Preprocess review text using vectorized operations
df['cleaned_description'] = df['review_description'].astype(str).str.lower().str.replace(r'[^\w\s]', '', regex=True)

# Optimize categorize_review logic
compiled_regexes = []
for category, keywords in bug_keywords.items():
    if category == "Other" or not keywords: # Skip "Other" or empty keyword lists for regex generation
        continue
    # Escape keywords to ensure they are treated as literal strings in regex
    pattern = r'|'.join([re.escape(kw) for kw in keywords])
    compiled_regexes.append((category, re.compile(pattern, flags=re.IGNORECASE)))

df['bug_category'] = "Other" # Default category

for category, regex_pattern in compiled_regexes:
    # Apply this regex to rows that are still "Other"
    # Ensure `na=False` for `str.contains` to treat NaN/empty strings in 'cleaned_description' as non-matches
    mask = (df['bug_category'] == "Other") & df['cleaned_description'].str.contains(regex_pattern, na=False, regex=True)
    df.loc[mask, 'bug_category'] = category

df[['review_description', 'bug_category', 'review_date', 'appVersion']].to_csv("categorized_bugs.csv", index=False)
