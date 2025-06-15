bug_keywords = {
    "Login Error": [
        "login", "log in", "sign in", "cannot access", "authentication", "credentials",
        "cannot log", "cannot login", "login problem", "login issue", "log my account",
        "log account", "unable to login", "cant login", "signing in", "invalid password",
        "username invalid"
    ],
    "Contact Change Issue": [
        "contact info", "phone number", "mobile number", "change number", "update contact",
        "contact information", "contact details", "phone details", "cannot change",
        "number change", "update number", "edit contact"
    ],
    "UI Issue": [
        "layout", "screen", "button", "display", "alignment", "responsive", "interface",
        "user interface", "ui", "blank screen", "design"
    ],
    "Crash/Freeze": [
        "crash", "crashed", "crashing", "freeze", "freezing", "stuck", "unresponsive",
        "not responding", "force close", "force stop", "hang", "stuck on"
    ],
    "Payment Issue": [
        "payment", "bill", "topup", "recharge", "transaction", "fail", "credit card",
        "unable to pay", "payment failed", "payment error", "bill pay", "transaction error",
        "session timeout"
    ],
    "Slow Performance": [
        "slow", "lag", "delay", "loading", "takes time", "wait", "performance",
        "slow loading", "lagging", "sluggish"
    ],
    "Notification Problem": [
        "notification", "alert", "reminder", "not getting", "missing notifications",
        "notification error", "no notification"
    ],
    "Update Issue": [
        "update", "updated", "version", "after update", "since update", "new version",
        "upgrade", "latest update", "post update"
    ],
    "Server Error": [
        "server error", "server down", "server not responding", "cannot connect",
        "connection error", "500", "503", "internal server error", "system error",
        "service unavailable"
    ],
    "Network/Connection": [
        "network", "connection", "wifi", "internet", "signal", "offline", "no service",
        "poor connection", "weak signal", "disconnected", "no internet", "network error",
        "connection lost"
    ],
    "OTP Issue": [
        "otp", "one time password", "verification code", "verification", "otp code",
        "didnt get otp", "otp not received"
    ],
    "Other": []
}

import pandas as pd
import re

# Load prioritized bug data
df = pd.read_csv("data/prioritized_bugs.csv")

# Preprocess review text
def preprocess(text):
    text = str(text).lower()
    text = text.replace("can't", "cannot")
    text = text.replace("can t", "cannot")
    text = re.sub(r'[^\w\s]', '', text)  # Remove punctuation
    return text

df['cleaned_description'] = df['review_description'].apply(preprocess)

def categorize_review(text):
    text = text.lower()
    # Flatten all keywords with their categories and sort by length (longest first)
    keyword_category_pairs = []
    for category, keywords in bug_keywords.items():
        for keyword in keywords:
            keyword_category_pairs.append((keyword.lower(), category))
    keyword_category_pairs.sort(key=lambda x: len(x[0]), reverse=True)

    for keyword, category in keyword_category_pairs:
        # Use word boundaries for all keywords (multi-word and single)
        pattern = r'\b' + re.escape(keyword) + r'\b'
        if re.search(pattern, text):
            return category
    return "Other"

df['bug_category'] = df['cleaned_description'].apply(categorize_review)

df[['review_description', 'bug_category', 'review_date', 'appVersion']].to_csv("data/categorized_bugs.csv", index=False)

# Debug Review Example
if __name__ == "__main__":
    sample_review = "cannot log complaints very bad service always not able to do an action"
    cleaned = preprocess(sample_review)
    category = categorize_review(cleaned)
    print(f"Sample review: {sample_review}")
    print(f"Predicted category: {category}")
